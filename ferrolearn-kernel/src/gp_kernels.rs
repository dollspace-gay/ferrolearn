//! Kernel functions for Gaussian Process regression and classification.
//!
//! This module provides the [`GPKernel`] trait and standard covariance kernels
//! used in Gaussian Process models. Unlike the NW/local-polynomial kernels in
//! [`crate::kernels`], GP kernels compute *pairwise covariance matrices*
//! between sets of input points.
//!
//! # Available kernels
//!
//! | Kernel | Formula |
//! |--------|---------|
//! | [`RBFKernel`] | `k(x, x') = exp(-||x - x'||^2 / (2 l^2))` |
//! | [`MaternKernel`] | Matern family (any nu via closed forms / Bessel K_ν / nu=inf→RBF) |
//! | [`RationalQuadratic`] | `k(x, x') = (1 + ||x - x'||² / (2αl²))^{-α}` |
//! | [`ExpSineSquared`] | `k(x, x') = exp(-2 sin²(π||x-x'||/p) / l²)` |
//! | [`ConstantKernel`] | `k(x, x') = c` |
//! | [`WhiteKernel`] | `k(x, x') = sigma^2 * delta(x, x')` |
//! | [`DotProductKernel`] | `k(x, x') = sigma_0^2 + x . x'` |
//! | [`SumKernel`] | `k = k1 + k2` |
//! | [`ProductKernel`] | `k = k1 * k2` |
//! | [`Exponentiation`] | `k = base^exponent` |
//!
//! Kernels can be composed via `+` and `*` operators on `Box<dyn GPKernel<F>>`.
//!
//! ## REQ status
//!
//! Mirrors `sklearn.gaussian_process.kernels` (`kernels.py`, v1.5.2 commit
//! 156ef14). Design doc: `.design/kernel/gp_kernels.md` (18 REQs). Every REQ is
//! BINARY (R-DEFER-2): SHIPPED or NOT-STARTED (with a concrete blocker). GP
//! kernels are DETERMINISTIC, so the existing kernels' matrix values are directly
//! oracle-verified element-wise (27 green guards in `tests/divergence_gp_kernels.rs`).
//! The gaps are missing-feature / prerequisite blockers, not value errors.
//!
//! **12 SHIPPED / 5 NOT-STARTED.**
//!
//! | REQ | Status | Notes |
//! |---|---|---|
//! | REQ-1 (`RBFKernel`) | SHIPPED | `exp(-‖x−x'‖²/(2·length_scale²))` matches `RBF(length_scale)` (`kernels.py:1445`); oracle element-wise <1e-12. |
//! | REQ-2 (`MaternKernel`, nu ∈ {0.5, 1.5, 2.5}) | SHIPPED | the three closed-form Matern formulas match `Matern(length_scale, nu)` (`:1598`) for the supported nu; oracle <1e-12. (General nu is REQ-10, now SHIPPED.) |
//! | REQ-3 (`ConstantKernel`) | SHIPPED | `k = constant_value` matches `ConstantKernel(constant_value)` (`:1184`); oracle. |
//! | REQ-4 (`DotProductKernel`) | SHIPPED | `k = sigma_0² + x·x'` matches `DotProduct(sigma_0)` (`:2099`); oracle. |
//! | REQ-5 (`SumKernel` / `ProductKernel`) | SHIPPED | `k1 ± k2` / `k1·k2` element-wise + diagonal match sklearn `Sum`/`Product` (`:796`,`:893`); theta concatenation order (k1 then k2) matches. |
//! | REQ-6 (log-space `get_params`/`set_params`) | SHIPPED | hyperparameters round-trip in log space, matching sklearn `theta` (e.g. `(RBF(1.5)+White(0.1)).theta == [ln1.5, ln0.1]`); oracle-verified. |
//! | REQ-7 (production consumer) | SHIPPED | `GaussianProcessRegressor`/`GaussianProcessClassifier` (`gaussian_process.rs`, `gp_classifier.rs`) consume `compute`/`diagonal` in `fit`/`predict`; re-exported via `lib.rs`. |
//! | REQ-8 (`eval_gradient` / dK/dθ) | NOT-STARTED | trait has `compute`/`diagonal` only; sklearn `__call__(eval_gradient=True)` returns `(K, K_gradient)` shape `(n,n,n_dims)` for the GPR LML optimizer. Blocker #1912. |
//! | REQ-9 (theta/bounds/Hyperparameter machinery) | NOT-STARTED | no `bounds`/`Hyperparameter`/`fixed`/`n_dims`; only flat log-space params (`:272-358`). Blocker #1913. |
//! | REQ-10 (Matern general nu) | SHIPPED | `MaternKernel::compute` now evaluates the modified-Bessel general formula `(2^{1-ν}/Γ(ν))·(√(2ν)·d)^ν·K_ν(√(2ν)·d)` for nu ∉ {0.5,1.5,2.5,inf} (`kernels.py:1729-1735`) and `nu=inf → exp(-d²/2)` (RBF, `:1727-1728`); `d≈0 → 1.0`. `K_ν` via `crate::bessel::bessel_k` (Numerical Recipes `bessik`, Temme series + CF2, ~1e-10 vs `scipy.special.kv`); Γ via `statrs::function::gamma::gamma`. Oracle `Matern(1.0,3.5)(X)[0,1]=0.5449424471128748` matches (no longer the RBF `0.6065`). Non-test consumer: `GaussianProcessRegressor` (`gaussian_process.rs` `fn fit`/`predict` via `kernel.compute`) — guarded by `tests/divergence_gaussian_process.rs::divergence_matern_general_nu_predict_std` (un-ignored). In-crate: `matern_general_nu_35_matches_sklearn`/`matern_general_nu_07_matches_sklearn`/`matern_nu_inf_is_rbf`/`matern_general_agrees_with_closed_forms`; `bessel::tests::kv_matches_scipy_oracle`. (Closes #1914, #2375.) |
//! | REQ-11 (`WhiteKernel` Y-given semantics) | NOT-STARTED | `compute(X,X)` row-equality → `noise·I` vs sklearn explicit-Y `zeros` (`:1416`); GPR relies on the `noise·I` self-path, so a faithful fix needs a Y=None channel rippling across kernels + GPR/GPC. Blocker #1915. |
//! | REQ-12 (anisotropic length_scale) | NOT-STARTED | scalar `length_scale` only; sklearn accepts a per-feature array (`:1472-1475`). Blocker #1916. |
//! | REQ-13 (`RationalQuadratic`) | SHIPPED | isotropic formula `(1 + d²/(2αl²))^{-α}` matches sklearn `RationalQuadratic` (`:1798`). |
//! | REQ-14 (constructor defaults / `Default`) | SHIPPED | `Default` impls call `new()` with sklearn's keyword defaults for RBF, Matern, RationalQuadratic, ExpSineSquared, Constant, White, and DotProduct; guarded by `gp_kernel_defaults_match_sklearn`. |
//! | REQ-15 (ferray substrate) | NOT-STARTED | `ndarray` arrays vs `ferray-core` (R-SUBSTRATE-1). Blocker #1919. |
//! | REQ-16 (`ExpSineSquared`) | SHIPPED | periodic formula `exp(-2 sin²(πd/p)/l²)` matches sklearn `ExpSineSquared` (`:1954`). |
//! | REQ-17 (`Exponentiation`) | SHIPPED | wraps any `GPKernel` and raises matrix/diagonal values to `exponent`, matching sklearn `Exponentiation` (`:993`). |
//! | REQ-18 (remaining missing kernel) | NOT-STARTED | no `CompoundKernel` (`:514`). |

use ndarray::{Array1, Array2};
use num_traits::Float;

/// Trait for covariance kernels used in Gaussian Process models.
///
/// A GP kernel computes the covariance between pairs of input points.
/// Implementations must be thread-safe (`Send + Sync`) and expose their
/// hyperparameters for optimization.
pub trait GPKernel<F: Float>: Send + Sync {
    /// Compute the full kernel matrix `K(X1, X2)` where `K[i,j] = k(x1_i, x2_j)`.
    ///
    /// # Arguments
    ///
    /// * `x1` - First set of points, shape `(n1, d)`.
    /// * `x2` - Second set of points, shape `(n2, d)`.
    ///
    /// # Returns
    ///
    /// Kernel matrix of shape `(n1, n2)`.
    fn compute(&self, x1: &Array2<F>, x2: &Array2<F>) -> Array2<F>;

    /// Compute only the diagonal of `K(X, X)`.
    ///
    /// This is more efficient than computing the full matrix when only
    /// variances are needed (e.g., predictive variance).
    fn diagonal(&self, x: &Array2<F>) -> Array1<F>;

    /// Number of tunable hyperparameters.
    fn n_params(&self) -> usize;

    /// Get the current hyperparameter values (in log space for positive params).
    fn get_params(&self) -> Vec<F>;

    /// Set hyperparameters from a slice (in log space for positive params).
    ///
    /// # Panics
    ///
    /// May panic if `params.len() != self.n_params()`.
    fn set_params(&mut self, params: &[F]);

    /// Clone this kernel into a boxed trait object.
    fn clone_box(&self) -> Box<dyn GPKernel<F>>;
}

impl<F: Float + Send + Sync + 'static> Clone for Box<dyn GPKernel<F>> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

// ---------------------------------------------------------------------------
// Squared-distance helper
// ---------------------------------------------------------------------------

/// Compute the squared Euclidean distance matrix between rows of `x1` and `x2`.
fn squared_distances<F: Float>(x1: &Array2<F>, x2: &Array2<F>) -> Array2<F> {
    let n1 = x1.nrows();
    let n2 = x2.nrows();
    let mut dists = Array2::<F>::zeros((n1, n2));
    for i in 0..n1 {
        for j in 0..n2 {
            let mut sum = F::zero();
            for d in 0..x1.ncols() {
                let diff = x1[[i, d]] - x2[[j, d]];
                sum = sum + diff * diff;
            }
            dists[[i, j]] = sum;
        }
    }
    dists
}

/// Compute the Euclidean distance matrix between rows of `x1` and `x2`.
fn euclidean_distances<F: Float>(x1: &Array2<F>, x2: &Array2<F>) -> Array2<F> {
    squared_distances(x1, x2).mapv(num_traits::Float::sqrt)
}

// ---------------------------------------------------------------------------
// RBF (Squared Exponential) Kernel
// ---------------------------------------------------------------------------

/// Radial Basis Function (squared exponential) kernel.
///
/// `k(x, x') = exp(-||x - x'||^2 / (2 * length_scale^2))`
///
/// This is the most commonly used GP kernel. It produces infinitely
/// differentiable (very smooth) functions.
#[derive(Debug, Clone)]
pub struct RBFKernel<F> {
    /// Length scale parameter. Controls the smoothness of the function.
    pub length_scale: F,
}

impl<F: Float> RBFKernel<F> {
    /// Create a new RBF kernel with the given length scale.
    #[must_use]
    pub fn new(length_scale: F) -> Self {
        Self { length_scale }
    }
}

impl<F: Float> Default for RBFKernel<F> {
    /// Mirrors sklearn `RBF(length_scale=1.0)` (`kernels.py:1508`).
    fn default() -> Self {
        Self::new(F::one())
    }
}

impl<F: Float + Send + Sync + 'static> GPKernel<F> for RBFKernel<F> {
    fn compute(&self, x1: &Array2<F>, x2: &Array2<F>) -> Array2<F> {
        let two = F::from(2.0).unwrap();
        let ls2 = self.length_scale * self.length_scale;
        let sq = squared_distances(x1, x2);
        sq.mapv(|d| (-d / (two * ls2)).exp())
    }

    fn diagonal(&self, x: &Array2<F>) -> Array1<F> {
        // k(x, x) = exp(0) = 1 for all x
        Array1::from_elem(x.nrows(), F::one())
    }

    fn n_params(&self) -> usize {
        1
    }

    fn get_params(&self) -> Vec<F> {
        vec![self.length_scale.ln()]
    }

    fn set_params(&mut self, params: &[F]) {
        self.length_scale = params[0].exp();
    }

    fn clone_box(&self) -> Box<dyn GPKernel<F>> {
        Box::new(self.clone())
    }
}

// ---------------------------------------------------------------------------
// Matern Kernel
// ---------------------------------------------------------------------------

/// Matern kernel with parameter `nu` controlling smoothness.
///
/// Any `nu > 0` is supported, matching `sklearn.gaussian_process.kernels.Matern`.
/// The half-integer cases use the analytic closed forms; `nu = inf` reduces to
/// the [`RBFKernel`]; every other `nu` evaluates the general modified-Bessel
/// formula (`kernels.py:1729-1735`).
///
/// - `0.5`: Exponential kernel (Ornstein-Uhlenbeck). Produces rough, non-differentiable paths.
/// - `1.5`: Once-differentiable functions. Good default for many applications.
/// - `2.5`: Twice-differentiable functions. Smoother than 1.5, less smooth than RBF.
/// - `inf`: equivalent to the RBF (squared-exponential) kernel.
///
/// Closed forms (`r = ||x - x'||`):
/// - nu = 0.5: `k(x,x') = exp(-r/l)`
/// - nu = 1.5: `k(x,x') = (1 + sqrt(3)*r/l) * exp(-sqrt(3)*r/l)`
/// - nu = 2.5: `k(x,x') = (1 + sqrt(5)*r/l + 5*r^2/(3*l^2)) * exp(-sqrt(5)*r/l)`
///
/// General nu (`d = r/l`, `K_ν` the modified Bessel function of the second kind):
/// `k(x,x') = (2^{1-ν}/Γ(ν)) · (sqrt(2ν)·d)^ν · K_ν(sqrt(2ν)·d)`, with the limit
/// `k → 1` as `d → 0`.
#[derive(Debug, Clone)]
pub struct MaternKernel<F> {
    /// Length scale parameter.
    pub length_scale: F,
    /// Smoothness parameter. Any `nu > 0` (or `inf` for the RBF limit).
    pub nu: F,
}

impl<F: Float> MaternKernel<F> {
    /// Create a new Matern kernel.
    ///
    /// # Arguments
    ///
    /// * `length_scale` - Length scale parameter (positive).
    /// * `nu` - Smoothness parameter. Any `nu > 0` (or `inf` for the RBF limit);
    ///   the half-integer values `0.5/1.5/2.5` use the closed forms, all other
    ///   `nu` use the general modified-Bessel formula.
    #[must_use]
    pub fn new(length_scale: F, nu: F) -> Self {
        Self { length_scale, nu }
    }
}

impl<F: Float> Default for MaternKernel<F> {
    /// Mirrors sklearn `Matern(length_scale=1.0, nu=1.5)` (`kernels.py:1678`).
    fn default() -> Self {
        Self::new(F::one(), F::from(1.5).unwrap_or_else(F::one))
    }
}

impl<F: Float + Send + Sync + 'static> GPKernel<F> for MaternKernel<F> {
    fn compute(&self, x1: &Array2<F>, x2: &Array2<F>) -> Array2<F> {
        let dists = euclidean_distances(x1, x2);
        let ls = self.length_scale;
        let half = F::from(0.5).unwrap();
        let one_point_five = F::from(1.5).unwrap();
        let two_point_five = F::from(2.5).unwrap();

        if (self.nu - half).abs() < F::from(1e-8).unwrap() {
            // nu = 0.5: exponential
            dists.mapv(|r| (-r / ls).exp())
        } else if (self.nu - one_point_five).abs() < F::from(1e-8).unwrap() {
            // nu = 1.5
            let sqrt3 = F::from(3.0f64.sqrt()).unwrap();
            dists.mapv(|r| {
                let z = sqrt3 * r / ls;
                (F::one() + z) * (-z).exp()
            })
        } else if (self.nu - two_point_five).abs() < F::from(1e-8).unwrap() {
            // nu = 2.5
            let sqrt5 = F::from(5.0f64.sqrt()).unwrap();
            let five_thirds = F::from(5.0 / 3.0).unwrap();
            dists.mapv(|r| {
                let z = sqrt5 * r / ls;
                let r_over_l = r / ls;
                (F::one() + z + five_thirds * r_over_l * r_over_l) * (-z).exp()
            })
        } else if self.nu.is_infinite() {
            // nu = inf: the RBF (squared-exponential) limit
            // (`kernels.py:1727-1728`: `K = exp(-(dists**2)/2)`).
            let two = F::one() + F::one();
            let ls2 = ls * ls;
            let sq = squared_distances(x1, x2);
            sq.mapv(|d| (-d / (two * ls2)).exp())
        } else {
            // General nu: the modified-Bessel Matern (`kernels.py:1729-1735`):
            //   K = (2^{1-ν}/Γ(ν)) · t^ν · K_ν(t),  t = √(2ν)·(d/l),
            // where d is the Euclidean distance and K_ν is the modified Bessel
            // function of the second kind (`scipy.special.kv` → `bessel_k`).
            // At d → 0 the kernel limit is 1 (sklearn adds `eps` to dodge the
            // 0·∞; we special-case d ≈ 0 → 1.0 directly).
            let nu = self.nu.to_f64().unwrap_or(f64::NAN);
            let gamma_nu = statrs::function::gamma::gamma(nu);
            let coef = 2.0_f64.powf(1.0 - nu) / gamma_nu;
            let sqrt_2nu = (2.0 * nu).sqrt();
            let eps = F::from(1e-12).unwrap_or_else(F::zero);
            dists.mapv(|r| {
                if r <= eps {
                    return F::one();
                }
                let d = (r / ls).to_f64().unwrap_or(f64::NAN);
                let t = sqrt_2nu * d;
                let val = coef * t.powf(nu) * crate::bessel::bessel_k(nu, t);
                F::from(val).unwrap_or_else(F::zero)
            })
        }
    }

    fn diagonal(&self, x: &Array2<F>) -> Array1<F> {
        // k(x, x) = 1 for all Matern kernels (distance 0)
        Array1::from_elem(x.nrows(), F::one())
    }

    fn n_params(&self) -> usize {
        1 // only length_scale is optimizable; nu is fixed
    }

    fn get_params(&self) -> Vec<F> {
        vec![self.length_scale.ln()]
    }

    fn set_params(&mut self, params: &[F]) {
        self.length_scale = params[0].exp();
    }

    fn clone_box(&self) -> Box<dyn GPKernel<F>> {
        Box::new(self.clone())
    }
}

// ---------------------------------------------------------------------------
// Rational Quadratic Kernel
// ---------------------------------------------------------------------------

/// Rational Quadratic covariance kernel.
///
/// This mirrors `sklearn.gaussian_process.kernels.RationalQuadratic`'s
/// isotropic variant:
///
/// `k(x, x') = (1 + ||x - x'||² / (2 * alpha * length_scale²))^{-alpha}`
///
/// `length_scale` controls the characteristic distance, and `alpha` controls
/// the relative weighting of large-scale and small-scale variation.
#[derive(Debug, Clone)]
pub struct RationalQuadratic<F> {
    /// Length scale parameter.
    pub length_scale: F,
    /// Scale-mixture parameter.
    pub alpha: F,
}

impl<F: Float> RationalQuadratic<F> {
    /// Create a new Rational Quadratic kernel.
    ///
    /// Mirrors sklearn's constructor argument order:
    /// `RationalQuadratic(length_scale=..., alpha=...)`.
    #[must_use]
    pub fn new(length_scale: F, alpha: F) -> Self {
        Self {
            length_scale,
            alpha,
        }
    }
}

impl<F: Float> Default for RationalQuadratic<F> {
    /// Mirrors sklearn `RationalQuadratic(length_scale=1.0, alpha=1.0)`.
    fn default() -> Self {
        Self::new(F::one(), F::one())
    }
}

impl<F: Float + Send + Sync + 'static> GPKernel<F> for RationalQuadratic<F> {
    fn compute(&self, x1: &Array2<F>, x2: &Array2<F>) -> Array2<F> {
        let two = F::from(2.0).unwrap();
        let ls2 = self.length_scale * self.length_scale;
        let denom = two * self.alpha * ls2;
        let sq = squared_distances(x1, x2);
        sq.mapv(|d| (F::one() + d / denom).powf(-self.alpha))
    }

    fn diagonal(&self, x: &Array2<F>) -> Array1<F> {
        Array1::from_elem(x.nrows(), F::one())
    }

    fn n_params(&self) -> usize {
        2
    }

    fn get_params(&self) -> Vec<F> {
        // sklearn theta order is alphabetical by hyperparameter name:
        // alpha, then length_scale.
        vec![self.alpha.ln(), self.length_scale.ln()]
    }

    fn set_params(&mut self, params: &[F]) {
        self.alpha = params[0].exp();
        self.length_scale = params[1].exp();
    }

    fn clone_box(&self) -> Box<dyn GPKernel<F>> {
        Box::new(self.clone())
    }
}

// ---------------------------------------------------------------------------
// Exp-Sine-Squared Kernel
// ---------------------------------------------------------------------------

/// Exp-Sine-Squared covariance kernel, also called the periodic kernel.
///
/// This mirrors `sklearn.gaussian_process.kernels.ExpSineSquared`'s isotropic
/// variant:
///
/// `k(x, x') = exp(-2 * sin(π * ||x - x'|| / periodicity)^2 / length_scale^2)`
#[derive(Debug, Clone)]
pub struct ExpSineSquared<F> {
    /// Length scale parameter.
    pub length_scale: F,
    /// Periodicity parameter.
    pub periodicity: F,
}

impl<F: Float> ExpSineSquared<F> {
    /// Create a new Exp-Sine-Squared kernel.
    #[must_use]
    pub fn new(length_scale: F, periodicity: F) -> Self {
        Self {
            length_scale,
            periodicity,
        }
    }
}

impl<F: Float> Default for ExpSineSquared<F> {
    /// Mirrors sklearn `ExpSineSquared(length_scale=1.0, periodicity=1.0)`.
    fn default() -> Self {
        Self::new(F::one(), F::one())
    }
}

impl<F: Float + Send + Sync + 'static> GPKernel<F> for ExpSineSquared<F> {
    fn compute(&self, x1: &Array2<F>, x2: &Array2<F>) -> Array2<F> {
        let two = F::from(2.0).unwrap();
        let pi = F::from(std::f64::consts::PI).unwrap();
        let ls2 = self.length_scale * self.length_scale;
        let dists = euclidean_distances(x1, x2);
        dists.mapv(|d| {
            let sin_arg = (pi * d / self.periodicity).sin();
            (-(two * sin_arg * sin_arg) / ls2).exp()
        })
    }

    fn diagonal(&self, x: &Array2<F>) -> Array1<F> {
        Array1::from_elem(x.nrows(), F::one())
    }

    fn n_params(&self) -> usize {
        2
    }

    fn get_params(&self) -> Vec<F> {
        vec![self.length_scale.ln(), self.periodicity.ln()]
    }

    fn set_params(&mut self, params: &[F]) {
        self.length_scale = params[0].exp();
        self.periodicity = params[1].exp();
    }

    fn clone_box(&self) -> Box<dyn GPKernel<F>> {
        Box::new(self.clone())
    }
}

// ---------------------------------------------------------------------------
// Constant Kernel
// ---------------------------------------------------------------------------

/// Constant kernel: `k(x, x') = constant_value`.
///
/// Useful as a component in composite kernels. When used as a product with
/// another kernel, it scales the signal variance.
#[derive(Debug, Clone)]
pub struct ConstantKernel<F> {
    /// The constant covariance value.
    pub constant_value: F,
}

impl<F: Float> ConstantKernel<F> {
    /// Create a new constant kernel.
    #[must_use]
    pub fn new(constant_value: F) -> Self {
        Self { constant_value }
    }
}

impl<F: Float> Default for ConstantKernel<F> {
    /// Mirrors sklearn `ConstantKernel(constant_value=1.0)` (`kernels.py:1233`).
    fn default() -> Self {
        Self::new(F::one())
    }
}

impl<F: Float + Send + Sync + 'static> GPKernel<F> for ConstantKernel<F> {
    fn compute(&self, x1: &Array2<F>, x2: &Array2<F>) -> Array2<F> {
        Array2::from_elem((x1.nrows(), x2.nrows()), self.constant_value)
    }

    fn diagonal(&self, x: &Array2<F>) -> Array1<F> {
        Array1::from_elem(x.nrows(), self.constant_value)
    }

    fn n_params(&self) -> usize {
        1
    }

    fn get_params(&self) -> Vec<F> {
        vec![self.constant_value.ln()]
    }

    fn set_params(&mut self, params: &[F]) {
        self.constant_value = params[0].exp();
    }

    fn clone_box(&self) -> Box<dyn GPKernel<F>> {
        Box::new(self.clone())
    }
}

// ---------------------------------------------------------------------------
// White Kernel
// ---------------------------------------------------------------------------

/// White (noise) kernel: `k(x, x') = noise_level * delta(x, x')`.
///
/// Adds independent identically distributed noise to the diagonal.
/// The covariance is `noise_level` when `x == x'` and zero otherwise.
///
/// In practice, during training the kernel matrix already includes the
/// noise on the diagonal, so this kernel only contributes to the diagonal
/// of `K(X_train, X_train)`.
#[derive(Debug, Clone)]
pub struct WhiteKernel<F> {
    /// Noise variance.
    pub noise_level: F,
}

impl<F: Float> WhiteKernel<F> {
    /// Create a new white noise kernel.
    #[must_use]
    pub fn new(noise_level: F) -> Self {
        Self { noise_level }
    }
}

impl<F: Float> Default for WhiteKernel<F> {
    /// Mirrors sklearn `WhiteKernel(noise_level=1.0)` (`kernels.py:1363`).
    fn default() -> Self {
        Self::new(F::one())
    }
}

impl<F: Float + Send + Sync + 'static> GPKernel<F> for WhiteKernel<F> {
    fn compute(&self, x1: &Array2<F>, x2: &Array2<F>) -> Array2<F> {
        let n1 = x1.nrows();
        let n2 = x2.nrows();
        let mut k = Array2::<F>::zeros((n1, n2));
        // Only add noise where points coincide (same index in training set).
        // For K(X_train, X_train), this means the diagonal.
        if n1 == n2 {
            let eps = F::from(1e-12).unwrap();
            for i in 0..n1 {
                // Check if the rows are identical
                let mut same = true;
                for d in 0..x1.ncols() {
                    if (x1[[i, d]] - x2[[i, d]]).abs() > eps {
                        same = false;
                        break;
                    }
                }
                if same {
                    k[[i, i]] = self.noise_level;
                }
            }
        }
        k
    }

    fn diagonal(&self, x: &Array2<F>) -> Array1<F> {
        Array1::from_elem(x.nrows(), self.noise_level)
    }

    fn n_params(&self) -> usize {
        1
    }

    fn get_params(&self) -> Vec<F> {
        vec![self.noise_level.ln()]
    }

    fn set_params(&mut self, params: &[F]) {
        self.noise_level = params[0].exp();
    }

    fn clone_box(&self) -> Box<dyn GPKernel<F>> {
        Box::new(self.clone())
    }
}

// ---------------------------------------------------------------------------
// Dot Product Kernel
// ---------------------------------------------------------------------------

/// Dot product (linear) kernel: `k(x, x') = sigma_0^2 + x . x'`.
///
/// This kernel is non-stationary (translation-variant). It corresponds
/// to Bayesian linear regression when `sigma_0 = 0`.
#[derive(Debug, Clone)]
pub struct DotProductKernel<F> {
    /// Inhomogeneity parameter. Controls the bias term.
    pub sigma_0: F,
}

impl<F: Float> DotProductKernel<F> {
    /// Create a new dot product kernel.
    #[must_use]
    pub fn new(sigma_0: F) -> Self {
        Self { sigma_0 }
    }
}

impl<F: Float> Default for DotProductKernel<F> {
    /// Mirrors sklearn `DotProduct(sigma_0=1.0)` (`kernels.py:2156`).
    fn default() -> Self {
        Self::new(F::one())
    }
}

impl<F: Float + Send + Sync + 'static> GPKernel<F> for DotProductKernel<F> {
    fn compute(&self, x1: &Array2<F>, x2: &Array2<F>) -> Array2<F> {
        let s0_sq = self.sigma_0 * self.sigma_0;
        let dot = x1.dot(&x2.t());
        dot.mapv(|v| v + s0_sq)
    }

    fn diagonal(&self, x: &Array2<F>) -> Array1<F> {
        let s0_sq = self.sigma_0 * self.sigma_0;
        let n = x.nrows();
        let mut diag = Array1::<F>::zeros(n);
        for i in 0..n {
            let row = x.row(i);
            diag[i] = row.dot(&row) + s0_sq;
        }
        diag
    }

    fn n_params(&self) -> usize {
        1
    }

    fn get_params(&self) -> Vec<F> {
        vec![self.sigma_0.ln()]
    }

    fn set_params(&mut self, params: &[F]) {
        self.sigma_0 = params[0].exp();
    }

    fn clone_box(&self) -> Box<dyn GPKernel<F>> {
        Box::new(self.clone())
    }
}

// ---------------------------------------------------------------------------
// Sum Kernel
// ---------------------------------------------------------------------------

/// Sum of two kernels: `k(x, x') = k1(x, x') + k2(x, x')`.
///
/// Used to combine independent signal components. For example,
/// `ConstantKernel(1.0) * RBFKernel(1.0) + WhiteKernel(0.1)` models
/// a smooth signal plus independent noise.
pub struct SumKernel<F: Float + Send + Sync + 'static> {
    /// First kernel.
    pub k1: Box<dyn GPKernel<F>>,
    /// Second kernel.
    pub k2: Box<dyn GPKernel<F>>,
}

impl<F: Float + Send + Sync + 'static> std::fmt::Debug for SumKernel<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SumKernel").finish_non_exhaustive()
    }
}

impl<F: Float + Send + Sync + 'static> SumKernel<F> {
    /// Create a sum of two kernels.
    pub fn new(k1: Box<dyn GPKernel<F>>, k2: Box<dyn GPKernel<F>>) -> Self {
        Self { k1, k2 }
    }
}

impl<F: Float + Send + Sync + 'static> GPKernel<F> for SumKernel<F> {
    fn compute(&self, x1: &Array2<F>, x2: &Array2<F>) -> Array2<F> {
        let m1 = self.k1.compute(x1, x2);
        let m2 = self.k2.compute(x1, x2);
        m1 + m2
    }

    fn diagonal(&self, x: &Array2<F>) -> Array1<F> {
        let d1 = self.k1.diagonal(x);
        let d2 = self.k2.diagonal(x);
        d1 + d2
    }

    fn n_params(&self) -> usize {
        self.k1.n_params() + self.k2.n_params()
    }

    fn get_params(&self) -> Vec<F> {
        let mut params = self.k1.get_params();
        params.extend(self.k2.get_params());
        params
    }

    fn set_params(&mut self, params: &[F]) {
        let n1 = self.k1.n_params();
        self.k1.set_params(&params[..n1]);
        self.k2.set_params(&params[n1..]);
    }

    fn clone_box(&self) -> Box<dyn GPKernel<F>> {
        Box::new(SumKernel {
            k1: self.k1.clone_box(),
            k2: self.k2.clone_box(),
        })
    }
}

// ---------------------------------------------------------------------------
// Product Kernel
// ---------------------------------------------------------------------------

/// Product of two kernels: `k(x, x') = k1(x, x') * k2(x, x')`.
///
/// Used to scale kernels. For example, `ConstantKernel(c) * RBFKernel(l)`
/// produces an RBF kernel with signal variance `c`.
pub struct ProductKernel<F: Float + Send + Sync + 'static> {
    /// First kernel.
    pub k1: Box<dyn GPKernel<F>>,
    /// Second kernel.
    pub k2: Box<dyn GPKernel<F>>,
}

impl<F: Float + Send + Sync + 'static> std::fmt::Debug for ProductKernel<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProductKernel").finish_non_exhaustive()
    }
}

impl<F: Float + Send + Sync + 'static> ProductKernel<F> {
    /// Create a product of two kernels.
    pub fn new(k1: Box<dyn GPKernel<F>>, k2: Box<dyn GPKernel<F>>) -> Self {
        Self { k1, k2 }
    }
}

impl<F: Float + Send + Sync + 'static> GPKernel<F> for ProductKernel<F> {
    fn compute(&self, x1: &Array2<F>, x2: &Array2<F>) -> Array2<F> {
        let m1 = self.k1.compute(x1, x2);
        let m2 = self.k2.compute(x1, x2);
        m1 * m2
    }

    fn diagonal(&self, x: &Array2<F>) -> Array1<F> {
        let d1 = self.k1.diagonal(x);
        let d2 = self.k2.diagonal(x);
        d1 * d2
    }

    fn n_params(&self) -> usize {
        self.k1.n_params() + self.k2.n_params()
    }

    fn get_params(&self) -> Vec<F> {
        let mut params = self.k1.get_params();
        params.extend(self.k2.get_params());
        params
    }

    fn set_params(&mut self, params: &[F]) {
        let n1 = self.k1.n_params();
        self.k1.set_params(&params[..n1]);
        self.k2.set_params(&params[n1..]);
    }

    fn clone_box(&self) -> Box<dyn GPKernel<F>> {
        Box::new(ProductKernel {
            k1: self.k1.clone_box(),
            k2: self.k2.clone_box(),
        })
    }
}

// ---------------------------------------------------------------------------
// Exponentiation Kernel
// ---------------------------------------------------------------------------

/// Exponentiation of a base kernel: `k(x, x') = base(x, x')^exponent`.
///
/// This mirrors `sklearn.gaussian_process.kernels.Exponentiation`. The exponent
/// itself is a structural parameter, not a log-space hyperparameter; `theta`
/// and `set_params` delegate to the wrapped kernel.
pub struct Exponentiation<F: Float + Send + Sync + 'static> {
    /// Base kernel.
    pub kernel: Box<dyn GPKernel<F>>,
    /// Scalar exponent applied to the base covariance.
    pub exponent: F,
}

impl<F: Float + Send + Sync + 'static> std::fmt::Debug for Exponentiation<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Exponentiation")
            .field("exponent", &self.exponent.to_f64())
            .finish_non_exhaustive()
    }
}

impl<F: Float + Send + Sync + 'static> Exponentiation<F> {
    /// Create an exponentiated kernel.
    pub fn new(kernel: Box<dyn GPKernel<F>>, exponent: F) -> Self {
        Self { kernel, exponent }
    }
}

impl<F: Float + Send + Sync + 'static> GPKernel<F> for Exponentiation<F> {
    fn compute(&self, x1: &Array2<F>, x2: &Array2<F>) -> Array2<F> {
        self.kernel
            .compute(x1, x2)
            .mapv(|value| value.powf(self.exponent))
    }

    fn diagonal(&self, x: &Array2<F>) -> Array1<F> {
        self.kernel
            .diagonal(x)
            .mapv(|value| value.powf(self.exponent))
    }

    fn n_params(&self) -> usize {
        self.kernel.n_params()
    }

    fn get_params(&self) -> Vec<F> {
        self.kernel.get_params()
    }

    fn set_params(&mut self, params: &[F]) {
        self.kernel.set_params(params);
    }

    fn clone_box(&self) -> Box<dyn GPKernel<F>> {
        Box::new(Exponentiation {
            kernel: self.kernel.clone_box(),
            exponent: self.exponent,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    fn make_x1() -> Array2<f64> {
        Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]).unwrap()
    }

    fn make_x2() -> Array2<f64> {
        Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap()
    }

    // --- RBF ---

    #[test]
    fn rbf_self_covariance_is_one() {
        let k = RBFKernel::new(1.0);
        let x = make_x1();
        let km = k.compute(&x, &x);
        for i in 0..x.nrows() {
            assert_abs_diff_eq!(km[[i, i]], 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn rbf_symmetry() {
        let k = RBFKernel::new(1.0);
        let x = make_x1();
        let km = k.compute(&x, &x);
        for i in 0..x.nrows() {
            for j in 0..x.nrows() {
                assert_abs_diff_eq!(km[[i, j]], km[[j, i]], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn rbf_diagonal() {
        let k = RBFKernel::new(1.0);
        let x = make_x1();
        let diag = k.diagonal(&x);
        assert_eq!(diag.len(), 3);
        for &d in &diag {
            assert_abs_diff_eq!(d, 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn rbf_length_scale_effect() {
        let x1 = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();
        let x2 = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();

        let k_short = RBFKernel::new(0.5);
        let k_long = RBFKernel::new(5.0);

        let v_short = k_short.compute(&x1, &x2)[[0, 0]];
        let v_long = k_long.compute(&x1, &x2)[[0, 0]];

        // Longer length scale => higher correlation at same distance
        assert!(v_long > v_short);
    }

    #[test]
    fn rbf_params_roundtrip() {
        let mut k = RBFKernel::new(2.0);
        let params = k.get_params();
        assert_eq!(params.len(), 1);
        assert_abs_diff_eq!(params[0], 2.0f64.ln(), epsilon = 1e-12);

        k.set_params(&[1.0f64.ln()]);
        assert_abs_diff_eq!(k.length_scale, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn rbf_cross_covariance() {
        let k = RBFKernel::new(1.0);
        let x1 = make_x1();
        let x2 = make_x2();
        let km = k.compute(&x1, &x2);
        assert_eq!(km.dim(), (3, 2));
        // k(origin, origin) = 1
        assert_abs_diff_eq!(km[[0, 0]], 1.0, epsilon = 1e-12);
    }

    // --- Matern ---

    #[test]
    fn matern_05_is_exponential() {
        let k = MaternKernel::new(1.0, 0.5);
        let x1 = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();
        let x2 = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        let v = k.compute(&x1, &x2)[[0, 0]];
        assert_abs_diff_eq!(v, (-1.0f64).exp(), epsilon = 1e-12);
    }

    #[test]
    fn matern_15_at_zero() {
        let k = MaternKernel::new(1.0, 1.5);
        let x = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();
        let km = k.compute(&x, &x);
        assert_abs_diff_eq!(km[[0, 0]], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn matern_25_at_zero() {
        let k = MaternKernel::new(1.0, 2.5);
        let x = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();
        let km = k.compute(&x, &x);
        assert_abs_diff_eq!(km[[0, 0]], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn matern_symmetry() {
        let k = MaternKernel::new(1.0, 1.5);
        let x = make_x1();
        let km = k.compute(&x, &x);
        for i in 0..x.nrows() {
            for j in 0..x.nrows() {
                assert_abs_diff_eq!(km[[i, j]], km[[j, i]], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn matern_diagonal() {
        let k = MaternKernel::new(1.0, 2.5);
        let x = make_x1();
        let diag = k.diagonal(&x);
        for &d in &diag {
            assert_abs_diff_eq!(d, 1.0, epsilon = 1e-12);
        }
    }

    /// General-nu Matern via the modified Bessel `K_ν` (REQ-10).
    /// Live sklearn 1.5.2 oracle (R-CHAR-3), `X=[[0,0],[1,0],[0,1]]` (== `make_x1`):
    /// `Matern(length_scale=1.0, nu=3.5)(X)` →
    /// `[[1, 0.5449424471128748, 0.5449424471128748],
    ///   [0.5449424471128748, 1, 0.3280670124332057],
    ///   [0.5449424471128748, 0.3280670124332057, 1]]`.
    /// Generated by:
    /// `python3 -c "import numpy as np; from sklearn.gaussian_process.kernels \
    ///   import Matern; X=np.array([[0.,0.],[1.,0.],[0.,1.]]); \
    ///   print(Matern(length_scale=1.0,nu=3.5)(X).tolist())"`.
    #[test]
    fn matern_general_nu_35_matches_sklearn() {
        let k = MaternKernel::new(1.0, 3.5);
        let x = make_x1();
        let km = k.compute(&x, &x);
        // diagonal = 1 (d=0 limit)
        for i in 0..3 {
            assert_abs_diff_eq!(km[[i, i]], 1.0, epsilon = 1e-12);
        }
        // off-diagonal: distance 1 -> 0.5449424471128748,
        // distance sqrt(2) -> 0.3280670124332057.
        assert_abs_diff_eq!(km[[0, 1]], 0.544_942_447_112_874_8, epsilon = 1e-9);
        assert_abs_diff_eq!(km[[0, 2]], 0.544_942_447_112_874_8, epsilon = 1e-9);
        assert_abs_diff_eq!(km[[1, 2]], 0.328_067_012_433_205_7, epsilon = 1e-9);
        // This is NOT the RBF value (would be 0.6065306597126334) — the silent
        // fallback is gone.
        assert!((km[[0, 1]] - 0.606_530_659_712_633_4).abs() > 1e-3);
    }

    /// Non-half-integer nu (`nu=0.7`) Matern via `K_ν`. Live sklearn oracle:
    /// `Matern(length_scale=1.0, nu=0.7)(X)[0,1] = 0.406181840375756`,
    /// `[1,2] = 0.26180486407843745`.
    #[test]
    fn matern_general_nu_07_matches_sklearn() {
        let k = MaternKernel::new(1.0, 0.7);
        let x = make_x1();
        let km = k.compute(&x, &x);
        assert_abs_diff_eq!(km[[0, 0]], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(km[[0, 1]], 0.406_181_840_375_756, epsilon = 1e-9);
        assert_abs_diff_eq!(km[[1, 2]], 0.261_804_864_078_437_45, epsilon = 1e-9);
    }

    /// nu = inf reduces to the RBF kernel (`kernels.py:1727-1728`).
    /// Live oracle: `Matern(length_scale=1.0, nu=np.inf)(X) == RBF(1.0)(X)`.
    #[test]
    fn matern_nu_inf_is_rbf() {
        let x = make_x1();
        let km = MaternKernel::new(1.0, f64::INFINITY).compute(&x, &x);
        let rbf = RBFKernel::new(1.0).compute(&x, &x);
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(km[[i, j]], rbf[[i, j]], epsilon = 1e-12);
            }
        }
    }

    /// Cross-check: the general-nu Bessel formula AGREES with the dedicated
    /// closed forms at `nu ∈ {1.5, 2.5}` (the closed forms ARE the analytic
    /// simplification of the Bessel formula for half-integer nu).
    #[test]
    fn matern_general_agrees_with_closed_forms() {
        use crate::bessel::bessel_k;
        let gamma = statrs::function::gamma::gamma;
        for &d in &[0.5_f64, 1.0, 2.0, 3.7] {
            // nu = 1.5 closed form: (1 + sqrt(3) d)·exp(-sqrt(3) d)
            let z3 = 3.0_f64.sqrt() * d;
            let closed_15 = (1.0 + z3) * (-z3).exp();
            let nu = 1.5;
            let t = (2.0 * nu).sqrt() * d;
            let gen_15 = 2.0_f64.powf(1.0 - nu) / gamma(nu) * t.powf(nu) * bessel_k(nu, t);
            assert!(
                (gen_15 - closed_15).abs() < 1e-9,
                "nu=1.5 d={d}: general={gen_15}, closed={closed_15}"
            );

            // nu = 2.5 closed form: (1 + sqrt(5) d + 5 d^2/3)·exp(-sqrt(5) d)
            let z5 = 5.0_f64.sqrt() * d;
            let closed_25 = (1.0 + z5 + 5.0 * d * d / 3.0) * (-z5).exp();
            let nu = 2.5;
            let t = (2.0 * nu).sqrt() * d;
            let gen_25 = 2.0_f64.powf(1.0 - nu) / gamma(nu) * t.powf(nu) * bessel_k(nu, t);
            assert!(
                (gen_25 - closed_25).abs() < 1e-9,
                "nu=2.5 d={d}: general={gen_25}, closed={closed_25}"
            );
        }
    }

    #[test]
    fn matern_params_roundtrip() {
        let mut k = MaternKernel::new(3.0, 1.5);
        let params = k.get_params();
        assert_eq!(params.len(), 1);
        assert_abs_diff_eq!(params[0], 3.0f64.ln(), epsilon = 1e-12);

        k.set_params(&[0.5f64.ln()]);
        assert_abs_diff_eq!(k.length_scale, 0.5, epsilon = 1e-12);
    }

    // --- RationalQuadratic ---

    #[test]
    fn rational_quadratic_default_matches_sklearn() {
        let k = RationalQuadratic::new(1.0, 1.0);
        let x = make_x1();
        let km = k.compute(&x, &x);
        assert_abs_diff_eq!(km[[0, 0]], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(km[[0, 1]], 2.0 / 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(km[[1, 2]], 0.5, epsilon = 1e-12);
    }

    #[test]
    fn rational_quadratic_cross_covariance() {
        let k = RationalQuadratic::new(1.3, 0.7);
        let km = k.compute(&make_x1(), &make_x2());
        assert_eq!(km.dim(), (3, 2));
        assert_abs_diff_eq!(km[[0, 0]], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(km[[0, 1]], 0.651_255_953_178_008_1, epsilon = 1e-12);
        assert_abs_diff_eq!(km[[1, 0]], 0.781_322_696_181_108_2, epsilon = 1e-12);
    }

    #[test]
    fn rational_quadratic_params_roundtrip() {
        let mut k = RationalQuadratic::new(1.3, 0.7);
        let params = k.get_params();
        assert_eq!(params.len(), 2);
        assert_abs_diff_eq!(params[0], 0.7f64.ln(), epsilon = 1e-12);
        assert_abs_diff_eq!(params[1], 1.3f64.ln(), epsilon = 1e-12);

        k.set_params(&[2.0f64.ln(), 0.5f64.ln()]);
        assert_abs_diff_eq!(k.alpha, 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(k.length_scale, 0.5, epsilon = 1e-12);
    }

    // --- ExpSineSquared ---

    #[test]
    fn exp_sine_squared_default_matches_sklearn() {
        let k = ExpSineSquared::new(1.0, 1.0);
        let x = make_x1();
        let km = k.compute(&x, &x);
        assert_abs_diff_eq!(km[[0, 0]], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(km[[0, 1]], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(km[[1, 2]], 0.155_950_569_259_009_13, epsilon = 1e-12);
    }

    #[test]
    fn exp_sine_squared_cross_covariance() {
        let k = ExpSineSquared::new(1.3, 2.0);
        let km = k.compute(&make_x1(), &make_x2());
        assert_eq!(km.dim(), (3, 2));
        assert_abs_diff_eq!(km[[0, 0]], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(km[[0, 1]], 0.472_714_571_288_163, epsilon = 1e-12);
        assert_abs_diff_eq!(km[[1, 0]], 0.306_225_980_058_042_4, epsilon = 1e-12);
    }

    #[test]
    fn exp_sine_squared_params_roundtrip() {
        let mut k = ExpSineSquared::new(1.3, 2.0);
        let params = k.get_params();
        assert_eq!(params.len(), 2);
        assert_abs_diff_eq!(params[0], 1.3f64.ln(), epsilon = 1e-12);
        assert_abs_diff_eq!(params[1], 2.0f64.ln(), epsilon = 1e-12);

        k.set_params(&[0.7f64.ln(), 1.5f64.ln()]);
        assert_abs_diff_eq!(k.length_scale, 0.7, epsilon = 1e-12);
        assert_abs_diff_eq!(k.periodicity, 1.5, epsilon = 1e-12);
    }

    // --- Constant ---

    #[test]
    fn constant_kernel() {
        let k = ConstantKernel::new(3.0);
        let x1 = make_x1();
        let x2 = make_x2();
        let km = k.compute(&x1, &x2);
        for &v in &km {
            assert_abs_diff_eq!(v, 3.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn constant_diagonal() {
        let k = ConstantKernel::new(2.5);
        let x = make_x1();
        let diag = k.diagonal(&x);
        for &d in &diag {
            assert_abs_diff_eq!(d, 2.5, epsilon = 1e-12);
        }
    }

    // --- White ---

    #[test]
    fn white_kernel_diagonal_only() {
        let k = WhiteKernel::new(0.1);
        let x = make_x1();
        let km = k.compute(&x, &x);
        for i in 0..x.nrows() {
            for j in 0..x.nrows() {
                if i == j {
                    assert_abs_diff_eq!(km[[i, j]], 0.1, epsilon = 1e-12);
                } else {
                    assert_abs_diff_eq!(km[[i, j]], 0.0, epsilon = 1e-12);
                }
            }
        }
    }

    #[test]
    fn white_kernel_cross_different_sizes() {
        let k = WhiteKernel::new(0.1);
        let x1 = make_x1(); // 3 rows
        let x2 = make_x2(); // 2 rows
        // Cross-covariance: different sizes, so all zeros
        let km = k.compute(&x1, &x2);
        assert_eq!(km.dim(), (3, 2));
        for &v in &km {
            assert_abs_diff_eq!(v, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn white_kernel_cross_same_size() {
        // When x1 == x2 (same points), diagonal gets noise
        let k = WhiteKernel::new(0.1);
        let x = make_x1(); // 3 rows
        let km = k.compute(&x, &x);
        for i in 0..3 {
            assert_abs_diff_eq!(km[[i, i]], 0.1, epsilon = 1e-12);
        }
    }

    #[test]
    fn white_diagonal() {
        let k = WhiteKernel::new(0.5);
        let x = make_x1();
        let diag = k.diagonal(&x);
        for &d in &diag {
            assert_abs_diff_eq!(d, 0.5, epsilon = 1e-12);
        }
    }

    // --- DotProduct ---

    #[test]
    fn dot_product_at_origin() {
        let k = DotProductKernel::new(1.0);
        let x = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
        let km = k.compute(&x, &x);
        // sigma_0^2 + 0 = 1
        assert_abs_diff_eq!(km[[0, 0]], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn dot_product_linear() {
        let k = DotProductKernel::new(0.0);
        let x1 = Array2::from_shape_vec((1, 1), vec![3.0]).unwrap();
        let x2 = Array2::from_shape_vec((1, 1), vec![4.0]).unwrap();
        let km = k.compute(&x1, &x2);
        assert_abs_diff_eq!(km[[0, 0]], 12.0, epsilon = 1e-12);
    }

    #[test]
    fn dot_product_diagonal() {
        let k = DotProductKernel::new(1.0);
        let x = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let diag = k.diagonal(&x);
        // [1^2 + 2^2 + 1, 3^2 + 4^2 + 1] = [6, 26]
        assert_abs_diff_eq!(diag[0], 6.0, epsilon = 1e-12);
        assert_abs_diff_eq!(diag[1], 26.0, epsilon = 1e-12);
    }

    // --- Sum ---

    #[test]
    fn sum_kernel() {
        let k = SumKernel::new(
            Box::new(ConstantKernel::new(1.0)),
            Box::new(ConstantKernel::new(2.0)),
        );
        let x = make_x1();
        let km = k.compute(&x, &x);
        for &v in &km {
            assert_abs_diff_eq!(v, 3.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn sum_kernel_params() {
        let k = SumKernel::new(
            Box::new(RBFKernel::new(1.0)),
            Box::new(WhiteKernel::new(0.1)),
        );
        assert_eq!(k.n_params(), 2);
        let params = k.get_params();
        assert_eq!(params.len(), 2);
    }

    #[test]
    fn sum_kernel_diagonal() {
        let k = SumKernel::new(
            Box::new(RBFKernel::new(1.0)),
            Box::new(WhiteKernel::new(0.5)),
        );
        let x = make_x1();
        let diag = k.diagonal(&x);
        for &d in &diag {
            assert_abs_diff_eq!(d, 1.5, epsilon = 1e-12);
        }
    }

    // --- Product ---

    #[test]
    fn product_kernel() {
        let k = ProductKernel::new(
            Box::new(ConstantKernel::new(2.0)),
            Box::new(ConstantKernel::new(3.0)),
        );
        let x = make_x1();
        let km = k.compute(&x, &x);
        for &v in &km {
            assert_abs_diff_eq!(v, 6.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn product_kernel_params() {
        let k = ProductKernel::new(
            Box::new(ConstantKernel::new(2.0)),
            Box::new(RBFKernel::new(1.0)),
        );
        assert_eq!(k.n_params(), 2);
    }

    #[test]
    fn product_kernel_scaling() {
        // ConstantKernel(c) * RBFKernel(l) should scale RBF output by c
        let c = 5.0;
        let l = 1.0;
        let k_rbf = RBFKernel::new(l);
        let k_scaled = ProductKernel::new(
            Box::new(ConstantKernel::new(c)),
            Box::new(RBFKernel::new(l)),
        );
        let x1 = make_x1();
        let x2 = make_x2();
        let km_rbf = k_rbf.compute(&x1, &x2);
        let km_scaled = k_scaled.compute(&x1, &x2);
        for i in 0..x1.nrows() {
            for j in 0..x2.nrows() {
                assert_abs_diff_eq!(km_scaled[[i, j]], c * km_rbf[[i, j]], epsilon = 1e-12);
            }
        }
    }

    // --- Exponentiation ---

    #[test]
    fn exponentiation_rbf_matches_sklearn() {
        let k = Exponentiation::new(Box::new(RBFKernel::new(1.5)), 2.0);
        let km = k.compute(&make_x1(), &make_x1());
        assert_abs_diff_eq!(km[[0, 0]], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(km[[0, 1]], 0.641_180_388_429_954_6, epsilon = 1e-12);
        assert_abs_diff_eq!(km[[1, 2]], 0.411_112_290_507_187_5, epsilon = 1e-12);
    }

    #[test]
    fn exponentiation_diagonal_and_params_delegate() {
        let mut k = Exponentiation::new(Box::new(RationalQuadratic::new(1.3, 0.7)), 0.5);
        let diag = k.diagonal(&make_x1());
        for &d in &diag {
            assert_abs_diff_eq!(d, 1.0, epsilon = 1e-12);
        }
        let params = k.get_params();
        assert_eq!(params.len(), 2);
        assert_abs_diff_eq!(params[0], 0.7f64.ln(), epsilon = 1e-12);
        assert_abs_diff_eq!(params[1], 1.3f64.ln(), epsilon = 1e-12);

        k.set_params(&[2.0f64.ln(), 0.5f64.ln()]);
        let params = k.get_params();
        assert_abs_diff_eq!(params[0], 2.0f64.ln(), epsilon = 1e-12);
        assert_abs_diff_eq!(params[1], 0.5f64.ln(), epsilon = 1e-12);
    }

    // --- Clone ---

    #[test]
    fn clone_box_preserves_params() {
        let k: Box<dyn GPKernel<f64>> = Box::new(RBFKernel::new(2.5));
        let k2 = k.clone_box();
        let x = make_x1();
        let km1 = k.compute(&x, &x);
        let km2 = k2.compute(&x, &x);
        for i in 0..x.nrows() {
            for j in 0..x.nrows() {
                assert_abs_diff_eq!(km1[[i, j]], km2[[i, j]], epsilon = 1e-12);
            }
        }
    }

    // --- Constructor defaults (REQ-14) ---

    #[test]
    fn gp_kernel_defaults_match_sklearn() {
        // Symbolic sklearn 1.5.2 keyword defaults (R-CHAR-3):
        // RBF(length_scale=1.0) kernels.py:1508
        assert_eq!(RBFKernel::<f64>::default().length_scale, 1.0);
        // Matern(length_scale=1.0, nu=1.5) kernels.py:1678
        assert_eq!(MaternKernel::<f64>::default().length_scale, 1.0);
        assert_eq!(MaternKernel::<f64>::default().nu, 1.5);
        // RationalQuadratic(length_scale=1.0, alpha=1.0) kernels.py:1859
        assert_eq!(RationalQuadratic::<f64>::default().length_scale, 1.0);
        assert_eq!(RationalQuadratic::<f64>::default().alpha, 1.0);
        // ExpSineSquared(length_scale=1.0, periodicity=1.0) kernels.py:2017
        assert_eq!(ExpSineSquared::<f64>::default().length_scale, 1.0);
        assert_eq!(ExpSineSquared::<f64>::default().periodicity, 1.0);
        // ConstantKernel(constant_value=1.0) kernels.py:1233
        assert_eq!(ConstantKernel::<f64>::default().constant_value, 1.0);
        // DotProduct(sigma_0=1.0) kernels.py:2156
        assert_eq!(DotProductKernel::<f64>::default().sigma_0, 1.0);
        // WhiteKernel(noise_level=1.0) kernels.py:1363
        assert_eq!(WhiteKernel::<f64>::default().noise_level, 1.0);
    }

    // --- f32 support ---

    #[test]
    fn rbf_f32() {
        let k = RBFKernel::new(1.0f32);
        let x = Array2::from_shape_vec((2, 1), vec![0.0f32, 1.0]).unwrap();
        let km = k.compute(&x, &x);
        assert_abs_diff_eq!(km[[0, 0]], 1.0f32, epsilon = 1e-6);
        assert!(km[[0, 1]] > 0.0f32);
        assert!(km[[0, 1]] < 1.0f32);
    }

    #[test]
    fn matern_f32() {
        let k = MaternKernel::new(1.0f32, 1.5);
        let x = Array2::from_shape_vec((2, 1), vec![0.0f32, 1.0]).unwrap();
        let km = k.compute(&x, &x);
        assert_abs_diff_eq!(km[[0, 0]], 1.0f32, epsilon = 1e-5);
    }

    // --- Positive semi-definiteness ---

    #[test]
    fn rbf_positive_semidefinite() {
        // For a valid kernel, x^T K x >= 0 for all x
        let k = RBFKernel::new(1.0);
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5],
        )
        .unwrap();
        let km = k.compute(&x, &x);
        let v = array![1.0, -1.0, 0.5, 0.3, -0.2];
        let vtk = km.dot(&v);
        let quad_form = v.dot(&vtk);
        assert!(
            quad_form >= -1e-10,
            "Quadratic form should be non-negative, got {quad_form}"
        );
    }

    #[test]
    fn matern_15_positive_semidefinite() {
        let k = MaternKernel::new(1.0, 1.5);
        let x = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 2.0, 3.0]).unwrap();
        let km = k.compute(&x, &x);
        let v = array![1.0, -1.0, 0.5, 0.3];
        let vtk = km.dot(&v);
        let quad_form = v.dot(&vtk);
        assert!(
            quad_form >= -1e-10,
            "Quadratic form should be non-negative, got {quad_form}"
        );
    }
}
