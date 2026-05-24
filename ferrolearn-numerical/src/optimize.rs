//! Optimization algorithms for smooth objectives.
//!
//! This module provides scipy.optimize equivalents for the ferrolearn ML
//! framework. Four optimizers are available:
//!
//! - **[`NewtonCG`]** — Truncated Newton with conjugate-gradient inner loop
//!   and backtracking line search. Good for large-scale smooth problems when
//!   Hessian-vector products are cheap.
//! - **[`TrustRegionNCG`]** — Trust-region Newton-CG using the Steihaug-Toint
//!   CG subproblem solver. More robust than line-search Newton-CG, especially
//!   near saddle points or in ill-conditioned regions.
//! - **[`Powell`]** — Direction-set method (Numerical Recipes §10.5) for
//!   derivative-free ND minimization. Good when the gradient is unavailable
//!   or expensive — e.g. objectives that wrap an external solve or a
//!   discrete-sample interpolant. Equivalent to
//!   `scipy.optimize.minimize(method='powell')`.
//! - **[`brent_bounded`]** — Brent's method for 1-D bounded minimization on
//!   an interval `[a, b]`. Combines golden-section search with parabolic
//!   interpolation for superlinear convergence. Equivalent to
//!   `scipy.optimize.minimize_scalar(method='bounded')`.
//!
//! Newton-CG and Trust-Region require a closure that returns the objective
//! value and gradient, plus a second closure for Hessian-vector products.
//! `Powell` requires only the objective.
//!
//! # Example
//!
//! ```
//! use ndarray::{array, Array1};
//! use ferrolearn_numerical::optimize::{NewtonCG, TrustRegionNCG};
//!
//! // Minimize f(x) = 0.5 * (x0^2 + 2*x1^2)
//! let fun_grad = |x: &Array1<f64>| {
//!     let f = 0.5 * (x[0] * x[0] + 2.0 * x[1] * x[1]);
//!     let g = array![x[0], 2.0 * x[1]];
//!     (f, g)
//! };
//! let hessp = |_x: &Array1<f64>, p: &Array1<f64>| {
//!     array![p[0], 2.0 * p[1]]
//! };
//!
//! let result = NewtonCG::new()
//!     .minimize(fun_grad, hessp, array![5.0, 3.0])
//!     .unwrap();
//! assert!(result.converged);
//! ```

use ndarray::Array1;

/// Result of an optimization run.
///
/// Contains the solution vector, objective value, gradient, iteration count,
/// and a flag indicating whether the optimizer converged to the requested
/// tolerance.
#[derive(Debug, Clone)]
pub struct OptimizeResult {
    /// The solution vector.
    pub x: Array1<f64>,
    /// The objective function value at the solution.
    pub fun: f64,
    /// The gradient at the solution.
    pub grad: Array1<f64>,
    /// Number of iterations performed.
    pub n_iter: usize,
    /// Whether the optimizer converged.
    pub converged: bool,
}

// ---------------------------------------------------------------------------
// Newton-CG (Truncated Newton with Conjugate Gradient)
// ---------------------------------------------------------------------------

/// Newton-CG optimizer with backtracking line search.
///
/// Uses conjugate gradient to approximately solve the Newton system
/// `H d = -g` at each step, followed by a backtracking line search with the
/// Armijo sufficient-decrease condition. The CG inner loop terminates early
/// when negative curvature is encountered, ensuring a descent direction.
///
/// # Builder
///
/// ```
/// use ferrolearn_numerical::optimize::NewtonCG;
///
/// let opt = NewtonCG::new()
///     .with_max_iter(500)
///     .with_tol(1e-10);
/// ```
pub struct NewtonCG {
    /// Maximum number of outer (Newton) iterations.
    pub max_iter: usize,
    /// Gradient norm convergence tolerance.
    pub tol: f64,
    /// Maximum number of CG iterations per Newton step.
    pub max_cg_iter: usize,
}

impl Default for NewtonCG {
    fn default() -> Self {
        Self::new()
    }
}

impl NewtonCG {
    /// Create a new `NewtonCG` optimizer with default settings.
    ///
    /// Defaults: `max_iter = 200`, `tol = 1e-8`, `max_cg_iter = 200`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_iter: 200,
            tol: 1e-8,
            max_cg_iter: 200,
        }
    }

    /// Set the maximum number of outer Newton iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the gradient norm convergence tolerance.
    #[must_use]
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the maximum number of CG iterations per Newton step.
    #[must_use]
    pub fn with_max_cg_iter(mut self, max_cg_iter: usize) -> Self {
        self.max_cg_iter = max_cg_iter;
        self
    }

    /// Minimize an unconstrained objective using Newton-CG.
    ///
    /// # Arguments
    ///
    /// - `fun_grad` — closure returning `(f(x), grad f(x))`.
    /// - `hessp` — closure returning the Hessian-vector product `H(x) p`.
    /// - `x0` — initial guess.
    ///
    /// # Returns
    ///
    /// An [`OptimizeResult`] on success, or an error message if the input is
    /// invalid (e.g., zero-length initial guess).
    ///
    /// # Errors
    ///
    /// Returns `Err` if `x0` has length zero.
    pub fn minimize<FG, HP>(
        &self,
        mut fun_grad: FG,
        mut hessp: HP,
        x0: Array1<f64>,
    ) -> Result<OptimizeResult, String>
    where
        FG: FnMut(&Array1<f64>) -> (f64, Array1<f64>),
        HP: FnMut(&Array1<f64>, &Array1<f64>) -> Array1<f64>,
    {
        let n = x0.len();
        if n == 0 {
            return Err("initial guess x0 must have at least one element".into());
        }

        let mut x = x0;
        let (mut f, mut g) = fun_grad(&x);

        for iter in 0..self.max_iter {
            let g_norm = norm(&g);
            if g_norm < self.tol {
                return Ok(OptimizeResult {
                    x,
                    fun: f,
                    grad: g,
                    n_iter: iter,
                    converged: true,
                });
            }

            // CG tolerance: use Eisenstat-Walker forcing term.
            let cg_tol = f64::min(0.5, g_norm.sqrt()) * g_norm;

            // Approximately solve H d = -g using CG.
            let d = cg_solve(&mut hessp, &x, &g, cg_tol, self.max_cg_iter, n);

            // Backtracking line search (Armijo condition).
            let dg = dot(&g, &d);
            if dg >= 0.0 {
                // CG gave a non-descent direction (should be rare); fall back
                // to steepest descent.
                let d_sd = &g * (-1.0);
                let alpha = backtracking_line_search(&mut fun_grad, &x, f, &g, &d_sd);
                x = &x + &(&d_sd * alpha);
            } else {
                let alpha = backtracking_line_search(&mut fun_grad, &x, f, &g, &d);
                x = &x + &(&d * alpha);
            }

            let (f_new, g_new) = fun_grad(&x);
            f = f_new;
            g = g_new;
        }

        let g_norm = norm(&g);
        Ok(OptimizeResult {
            x,
            fun: f,
            grad: g,
            n_iter: self.max_iter,
            converged: g_norm < self.tol,
        })
    }
}

/// CG inner solver: approximately solve `H d = -g`.
///
/// Returns the step `d`. Terminates early on negative curvature or when the
/// residual is small enough.
fn cg_solve<HP>(
    hessp: &mut HP,
    x: &Array1<f64>,
    g: &Array1<f64>,
    tol: f64,
    max_iter: usize,
    _n: usize,
) -> Array1<f64>
where
    HP: FnMut(&Array1<f64>, &Array1<f64>) -> Array1<f64>,
{
    let n = g.len();
    let mut d = Array1::zeros(n);
    let mut r = g.clone(); // residual = g + H d = g (since d=0)
    let mut p = &r * (-1.0); // search direction

    for _cg_iter in 0..max_iter {
        let hp = hessp(x, &p);
        let p_hp = dot(&p, &hp);

        // Negative curvature: return current d if nonzero, else -g direction.
        if p_hp <= 1e-30 {
            if dot(&d, &d) > 0.0 {
                return d;
            }
            return g * (-1.0);
        }

        let r_dot_r = dot(&r, &r);
        let alpha = r_dot_r / p_hp;

        d = &d + &(&p * alpha);
        r = &r + &(&hp * alpha);

        let r_norm = norm(&r);
        if r_norm < tol {
            return d;
        }

        let r_dot_r_new = dot(&r, &r);
        let beta = r_dot_r_new / r_dot_r;
        p = &r * (-1.0) + &(&p * beta);
    }

    d
}

/// Backtracking line search with Armijo sufficient-decrease condition.
///
/// Returns a step size `alpha` satisfying `f(x + alpha d) <= f(x) + c alpha g^T d`.
fn backtracking_line_search<FG>(
    fun_grad: &mut FG,
    x: &Array1<f64>,
    f0: f64,
    g: &Array1<f64>,
    d: &Array1<f64>,
) -> f64
where
    FG: FnMut(&Array1<f64>) -> (f64, Array1<f64>),
{
    let c = 1e-4;
    let rho = 0.5;
    let dg = dot(g, d);
    let mut alpha = 1.0;

    for _ in 0..40 {
        let x_new = x + &(d * alpha);
        let (f_new, _) = fun_grad(&x_new);
        if f_new <= f0 + c * alpha * dg {
            return alpha;
        }
        alpha *= rho;
    }

    alpha
}

// ---------------------------------------------------------------------------
// Trust-Region Newton-CG (Steihaug-Toint)
// ---------------------------------------------------------------------------

/// Trust-region Newton-CG optimizer using the Steihaug-Toint CG subproblem
/// solver.
///
/// At each iteration, the CG inner loop approximately solves `H d = -g`
/// subject to the constraint `||d|| <= delta` (the trust-region radius).
/// The trust-region radius is adapted based on the ratio of actual to
/// predicted reduction.
///
/// This method is more robust than line-search Newton-CG, especially for
/// problems with near-singular Hessians or saddle points.
///
/// # Builder
///
/// ```
/// use ferrolearn_numerical::optimize::TrustRegionNCG;
///
/// let opt = TrustRegionNCG::new()
///     .with_max_iter(500)
///     .with_tol(1e-10);
/// ```
pub struct TrustRegionNCG {
    /// Maximum number of outer iterations.
    pub max_iter: usize,
    /// Gradient norm convergence tolerance.
    pub tol: f64,
    /// Initial trust-region radius.
    pub initial_radius: f64,
    /// Maximum trust-region radius.
    pub max_radius: f64,
}

impl Default for TrustRegionNCG {
    fn default() -> Self {
        Self::new()
    }
}

impl TrustRegionNCG {
    /// Create a new `TrustRegionNCG` optimizer with default settings.
    ///
    /// Defaults: `max_iter = 200`, `tol = 1e-8`, `initial_radius = 1.0`,
    /// `max_radius = 1000.0`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_iter: 200,
            tol: 1e-8,
            initial_radius: 1.0,
            max_radius: 1000.0,
        }
    }

    /// Set the maximum number of outer iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the gradient norm convergence tolerance.
    #[must_use]
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the initial trust-region radius.
    #[must_use]
    pub fn with_initial_radius(mut self, radius: f64) -> Self {
        self.initial_radius = radius;
        self
    }

    /// Set the maximum trust-region radius.
    #[must_use]
    pub fn with_max_radius(mut self, radius: f64) -> Self {
        self.max_radius = radius;
        self
    }

    /// Minimize an unconstrained objective using trust-region Newton-CG
    /// (Steihaug-Toint).
    ///
    /// # Arguments
    ///
    /// - `fun_grad` — closure returning `(f(x), grad f(x))`.
    /// - `hessp` — closure returning the Hessian-vector product `H(x) p`.
    /// - `x0` — initial guess.
    ///
    /// # Returns
    ///
    /// An [`OptimizeResult`] on success, or an error message if the input is
    /// invalid (e.g., zero-length initial guess).
    ///
    /// # Errors
    ///
    /// Returns `Err` if `x0` has length zero.
    pub fn minimize<FG, HP>(
        &self,
        mut fun_grad: FG,
        mut hessp: HP,
        x0: Array1<f64>,
    ) -> Result<OptimizeResult, String>
    where
        FG: FnMut(&Array1<f64>) -> (f64, Array1<f64>),
        HP: FnMut(&Array1<f64>, &Array1<f64>) -> Array1<f64>,
    {
        let n = x0.len();
        if n == 0 {
            return Err("initial guess x0 must have at least one element".into());
        }

        let eta = 1e-4; // step acceptance threshold
        let mut x = x0;
        let (mut f, mut g) = fun_grad(&x);
        let mut delta = self.initial_radius;

        for iter in 0..self.max_iter {
            let g_norm = norm(&g);
            if g_norm < self.tol {
                return Ok(OptimizeResult {
                    x,
                    fun: f,
                    grad: g,
                    n_iter: iter,
                    converged: true,
                });
            }

            // Solve the trust-region CG subproblem (Steihaug-Toint).
            let d = steihaug_cg(&mut hessp, &x, &g, delta, n);

            // Predicted reduction: -( g^T d + 0.5 d^T H d ).
            let hd = hessp(&x, &d);
            let pred = -(dot(&g, &d) + 0.5 * dot(&d, &hd));

            // Actual reduction.
            let x_new = &x + &d;
            let (f_new, g_new) = fun_grad(&x_new);
            let ared = f - f_new;

            let d_norm = norm(&d);

            // Compute ratio rho = actual / predicted.
            let rho = if pred.abs() < 1e-30 {
                // Predicted reduction essentially zero — treat as good step
                // if actual reduction is non-negative.
                if ared >= 0.0 { 1.0 } else { 0.0 }
            } else {
                ared / pred
            };

            // Update trust-region radius.
            if rho < 0.25 {
                delta *= 0.25;
            } else if rho > 0.75 && (d_norm - delta).abs() < 1e-12 * delta.max(1.0) {
                delta = (2.0 * delta).min(self.max_radius);
            }

            // Accept or reject step.
            if rho > eta {
                x = x_new;
                f = f_new;
                g = g_new;
            }
        }

        let g_norm = norm(&g);
        Ok(OptimizeResult {
            x,
            fun: f,
            grad: g,
            n_iter: self.max_iter,
            converged: g_norm < self.tol,
        })
    }
}

/// Steihaug-Toint CG subproblem solver.
///
/// Approximately solves `H d = -g` subject to `||d|| <= delta`.
/// On negative curvature or if the CG step would leave the trust region,
/// the solution is extended (or truncated) to the trust-region boundary.
fn steihaug_cg<HP>(
    hessp: &mut HP,
    x: &Array1<f64>,
    g: &Array1<f64>,
    delta: f64,
    n: usize,
) -> Array1<f64>
where
    HP: FnMut(&Array1<f64>, &Array1<f64>) -> Array1<f64>,
{
    let max_cg = n.max(200);
    let tol = f64::min(0.5, norm(g).sqrt()) * norm(g);

    let mut d = Array1::zeros(n);
    let mut r = g.clone();
    let mut p = &r * (-1.0);

    // If gradient is already tiny, return zero step.
    if norm(&r) < tol {
        return d;
    }

    for _cg_iter in 0..max_cg {
        let hp = hessp(x, &p);
        let p_hp = dot(&p, &hp);

        // Negative curvature: go to trust-region boundary along p.
        if p_hp <= 1e-30 {
            let tau = boundary_step(&d, &p, delta);
            return &d + &(&p * tau);
        }

        let r_dot_r = dot(&r, &r);
        let alpha = r_dot_r / p_hp;

        let d_next = &d + &(&p * alpha);

        // If the CG step leaves the trust region, truncate to boundary.
        if norm(&d_next) >= delta {
            let tau = boundary_step(&d, &p, delta);
            return &d + &(&p * tau);
        }

        d = d_next;
        r = &r + &(&hp * alpha);

        if norm(&r) < tol {
            return d;
        }

        let r_dot_r_new = dot(&r, &r);
        let beta = r_dot_r_new / r_dot_r;
        p = &r * (-1.0) + &(&p * beta);
    }

    d
}

/// Find the positive scalar `tau` such that `||d + tau p|| = delta`.
///
/// Solves the quadratic `||d + tau p||^2 = delta^2` and returns the larger
/// (positive) root.
fn boundary_step(d: &Array1<f64>, p: &Array1<f64>, delta: f64) -> f64 {
    let dd = dot(d, d);
    let dp = dot(d, p);
    let pp = dot(p, p);

    // Quadratic: pp tau^2 + 2 dp tau + (dd - delta^2) = 0
    let discrim = dp * dp - pp * (dd - delta * delta);
    let sqrt_discrim = if discrim > 0.0 { discrim.sqrt() } else { 0.0 };

    // We want the positive root.
    (-dp + sqrt_discrim) / pp
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Euclidean (L2) norm of a vector.
#[inline]
fn norm(v: &Array1<f64>) -> f64 {
    dot(v, v).sqrt()
}

/// Dot product of two vectors.
#[inline]
fn dot(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.dot(b)
}

// ---------------------------------------------------------------------------
// Powell direction-set method (derivative-free ND minimization)
// ---------------------------------------------------------------------------

/// Powell's direction-set method — derivative-free ND minimization.
///
/// Equivalent to `scipy.optimize.minimize(method='powell')`. Given a
/// starting point and the canonical basis as the initial direction set,
/// repeatedly line-minimize along each direction; after every full sweep
/// of N line searches, evaluate the extrapolation test (Numerical
/// Recipes §10.5) and, if it passes, replace the direction of greatest
/// decrease with the net displacement vector. Iterate until the relative
/// function improvement falls below tolerance.
///
/// Line search is by golden-section after a tripling-step bracket — a
/// simple, robust strategy that gives reliable convergence on smooth
/// objectives. The closure is `FnMut`, so callers that hold state (e.g.
/// counters, caches) can borrow mutably across evaluations.
///
/// # Builder
///
/// ```
/// use ferrolearn_numerical::optimize::Powell;
///
/// let opt = Powell::new()
///     .with_max_iter(500)
///     .with_ftol(1e-10)
///     .with_initial_step(0.25);
/// ```
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use ferrolearn_numerical::optimize::Powell;
///
/// // Minimize f(x) = (x0 - 3)^2 + (x1 + 4)^2
/// let f = |x: &ndarray::Array1<f64>| (x[0] - 3.0).powi(2) + (x[1] + 4.0).powi(2);
/// let result = Powell::new().minimize(f, array![0.0, 0.0]);
/// assert!(result.converged);
/// assert!((result.x[0] - 3.0).abs() < 1e-4);
/// assert!((result.x[1] + 4.0).abs() < 1e-4);
/// ```
///
/// # Result
///
/// Powell is derivative-free; the returned [`OptimizeResult::grad`] is
/// filled with zeros (the algorithm never computes a gradient). Consumers
/// that need gradient information should use [`NewtonCG`] or
/// [`TrustRegionNCG`].
pub struct Powell {
    /// Maximum number of outer iterations (direction-set sweeps).
    pub max_iter: usize,
    /// Relative tolerance on objective improvement between sweeps.
    pub ftol: f64,
    /// Initial step size used to seed the line bracket.
    pub initial_step: f64,
    /// Maximum golden-section iterations per line search.
    pub line_max_iter: usize,
    /// Absolute x-tolerance for the line search.
    pub line_xtol: f64,
    /// Maximum bracketing iterations per line search before falling back
    /// to a fixed bracket.
    pub bracket_max_iter: usize,
}

impl Default for Powell {
    fn default() -> Self {
        Self::new()
    }
}

impl Powell {
    /// Create a new `Powell` optimizer with default settings.
    ///
    /// Defaults: `max_iter = 200`, `ftol = 1e-8`, `initial_step = 0.5`,
    /// `line_max_iter = 200`, `line_xtol = 1e-7`, `bracket_max_iter = 60`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_iter: 200,
            ftol: 1.0e-8,
            initial_step: 0.5,
            line_max_iter: 200,
            line_xtol: 1.0e-7,
            bracket_max_iter: 60,
        }
    }

    /// Set the maximum number of outer iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the relative tolerance on objective improvement.
    #[must_use]
    pub fn with_ftol(mut self, ftol: f64) -> Self {
        self.ftol = ftol;
        self
    }

    /// Set the initial step size for the line bracket.
    #[must_use]
    pub fn with_initial_step(mut self, initial_step: f64) -> Self {
        self.initial_step = initial_step;
        self
    }

    /// Set the maximum golden-section iterations per line search.
    #[must_use]
    pub fn with_line_max_iter(mut self, line_max_iter: usize) -> Self {
        self.line_max_iter = line_max_iter;
        self
    }

    /// Set the absolute x-tolerance for the line search.
    #[must_use]
    pub fn with_line_xtol(mut self, line_xtol: f64) -> Self {
        self.line_xtol = line_xtol;
        self
    }

    /// Set the maximum bracketing iterations before falling back.
    #[must_use]
    pub fn with_bracket_max_iter(mut self, bracket_max_iter: usize) -> Self {
        self.bracket_max_iter = bracket_max_iter;
        self
    }

    /// Minimize the objective `f` starting from `x0`.
    ///
    /// The initial direction set is the canonical basis (one unit vector
    /// per dimension of `x0`).
    pub fn minimize<F>(&self, mut f: F, x0: Array1<f64>) -> OptimizeResult
    where
        F: FnMut(&Array1<f64>) -> f64,
    {
        let n = x0.len();

        let mut x = x0;
        let mut dirs: Vec<Array1<f64>> = (0..n)
            .map(|i| {
                let mut v = Array1::<f64>::zeros(n);
                v[i] = 1.0;
                v
            })
            .collect();

        let mut fx = f(&x);
        let mut converged = false;
        let mut iterations = 0usize;

        for it in 0..self.max_iter {
            iterations = it + 1;
            let fx_start = fx;
            let x_start = x.clone();
            let mut largest_decrease = 0.0;
            let mut largest_decrease_idx = 0usize;

            for (i, dir) in dirs.clone().into_iter().enumerate() {
                let f_before = fx;
                let (alpha, f_after) = line_minimise_powell(&mut f, &x, &dir, self);
                for (xk, dk) in x.iter_mut().zip(dir.iter()) {
                    *xk += alpha * dk;
                }
                fx = f_after;
                let dec = f_before - f_after;
                if dec > largest_decrease {
                    largest_decrease = dec;
                    largest_decrease_idx = i;
                }
            }

            // Relative-improvement convergence test (Numerical Recipes §10.5).
            let denom = (fx_start.abs() + fx.abs()).max(1.0e-30);
            if 2.0 * (fx_start - fx).abs() <= self.ftol * denom {
                converged = true;
                break;
            }

            // Net displacement across the sweep + extrapolation step.
            let delta: Array1<f64> = &x - &x_start;
            let x_extrap: Array1<f64> = 2.0 * &x - &x_start;
            let f_extrap = f(&x_extrap);
            if f_extrap < fx_start {
                let two_term = 2.0 * (fx_start - 2.0 * fx + f_extrap);
                let lhs = fx_start - fx - largest_decrease;
                let test = two_term * lhs * lhs - largest_decrease * (fx_start - f_extrap).powi(2);
                if test < 0.0 {
                    let (alpha, f_after) = line_minimise_powell(&mut f, &x, &delta, self);
                    for (xk, dk) in x.iter_mut().zip(delta.iter()) {
                        *xk += alpha * dk;
                    }
                    fx = f_after;
                    // Replace the direction of greatest decrease with the
                    // net displacement; rotate the last basis vector into
                    // its slot so the direction set stays linearly
                    // independent (the canonical NR trick).
                    dirs[largest_decrease_idx] = dirs[n - 1].clone();
                    dirs[n - 1] = delta;
                }
            }
        }

        OptimizeResult {
            x,
            fun: fx,
            grad: Array1::<f64>::zeros(n),
            n_iter: iterations,
            converged,
        }
    }
}

/// 1-D line minimisation along `direction` starting at `origin`. Returns
/// `(alpha, f(origin + alpha * direction))`. Used internally by [`Powell`].
fn line_minimise_powell<F>(
    f: &mut F,
    origin: &Array1<f64>,
    direction: &Array1<f64>,
    cfg: &Powell,
) -> (f64, f64)
where
    F: FnMut(&Array1<f64>) -> f64,
{
    let mut probe = origin.clone();
    let mut g = |alpha: f64| -> f64 {
        for ((p, o), d) in probe.iter_mut().zip(origin.iter()).zip(direction.iter()) {
            *p = o + alpha * d;
        }
        f(&probe)
    };

    let (a, b, c) = bracket_min_powell(&mut g, 0.0, cfg.initial_step, cfg.bracket_max_iter);
    golden_section_powell(&mut g, a, b, c, cfg.line_xtol, cfg.line_max_iter)
}

/// Bracket a 1-D minimum by stepping from `ax` and then golden-ratio
/// expanding the upper bound until `f(b) < f(c)`. Falls back to a fixed
/// symmetric bracket around the origin if anything goes non-finite or
/// the iteration limit is exceeded. Used internally by [`Powell`].
fn bracket_min_powell<G>(g: &mut G, ax: f64, step: f64, bracket_max_iter: usize) -> (f64, f64, f64)
where
    G: FnMut(f64) -> f64,
{
    let gold: f64 = 1.618_033_988_749_895;
    let mut a = ax;
    let mut b = ax + step;
    let mut fa = g(a);
    let mut fb = g(b);
    if fb > fa {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }
    let mut c = b + gold * (b - a);
    let mut fc = g(c);
    let mut k = 0usize;
    while fb > fc && k < bracket_max_iter {
        a = b;
        fa = fb;
        b = c;
        fb = fc;
        c = b + gold * (b - a);
        fc = g(c);
        k += 1;
    }
    if !a.is_finite() || !b.is_finite() || !c.is_finite() {
        return (-step, 0.0, step);
    }
    let _ = (fa, fc);
    (a, b, c)
}

/// Golden-section search inside the bracket `(a, b, c)`. Returns
/// `(xmin, f(xmin))`. Used internally by [`Powell`].
fn golden_section_powell<G>(
    g: &mut G,
    a: f64,
    b: f64,
    c: f64,
    xtol: f64,
    max_iter: usize,
) -> (f64, f64)
where
    G: FnMut(f64) -> f64,
{
    let r = 0.618_033_988_749_895_f64;
    let mut x0 = a.min(c);
    let mut x3 = a.max(c);
    let mut x1;
    let mut x2;
    if (x3 - b).abs() > (b - x0).abs() {
        x1 = b;
        x2 = b + (1.0 - r) * (x3 - b);
    } else {
        x2 = b;
        x1 = b - (1.0 - r) * (b - x0);
    }
    let mut f1 = g(x1);
    let mut f2 = g(x2);
    for _ in 0..max_iter {
        if (x3 - x0).abs() < xtol * (x1.abs() + x2.abs()) + xtol {
            break;
        }
        if f2 < f1 {
            x0 = x1;
            x1 = x2;
            x2 = r * x1 + (1.0 - r) * x3;
            f1 = f2;
            f2 = g(x2);
        } else {
            x3 = x2;
            x2 = x1;
            x1 = r * x2 + (1.0 - r) * x0;
            f2 = f1;
            f1 = g(x1);
        }
    }
    if f1 < f2 { (x1, f1) } else { (x2, f2) }
}

// ---------------------------------------------------------------------------
// Brent's method for 1-D bounded minimization
// ---------------------------------------------------------------------------

/// Result of a 1-D minimization.
///
/// Contains the minimizer, function value at the minimizer, the number of
/// function evaluations, and a flag indicating convergence.
#[derive(Debug, Clone)]
pub struct Minimize1DResult {
    /// The minimizer x*.
    pub x: f64,
    /// The function value f(x*).
    pub fun: f64,
    /// Number of function evaluations.
    pub nfev: usize,
    /// Whether the optimization converged.
    pub success: bool,
}

/// Minimize a scalar function on the interval `[a, b]` using Brent's method.
///
/// Combines golden-section search with parabolic interpolation for
/// superlinear convergence. Matches
/// `scipy.optimize.minimize_scalar(method='bounded')`.
///
/// # Arguments
///
/// - `f` — the scalar objective function to minimize.
/// - `a` — lower bound of the search interval.
/// - `b` — upper bound of the search interval.
/// - `tol` — convergence tolerance on the bracket width.
/// - `max_iter` — maximum number of iterations.
///
/// # Panics
///
/// Does not panic. If `a >= b`, the function still proceeds but the result
/// may be trivial.
///
/// # Example
///
/// ```
/// use ferrolearn_numerical::optimize::brent_bounded;
///
/// let result = brent_bounded(|x| x * x, -1.0, 2.0, 1e-10, 500);
/// assert!(result.success);
/// assert!((result.x).abs() < 1e-8);
/// ```
#[must_use]
pub fn brent_bounded<F>(f: F, a: f64, b: f64, tol: f64, max_iter: usize) -> Minimize1DResult
where
    F: Fn(f64) -> f64,
{
    // Golden ratio constant: (3 - sqrt(5)) / 2 ≈ 0.381966
    let golden = 0.5 * (3.0 - 5.0_f64.sqrt());

    let (mut lo, mut hi) = if a < b { (a, b) } else { (b, a) };

    // x is the point with the least function value found so far.
    // w is the point with the second least value.
    // v is the previous value of w.
    let mut x = lo + golden * (hi - lo);
    let mut fx = f(x);
    let mut nfev = 1_usize;

    let mut w = x;
    let mut fw = fx;
    let mut v = x;
    let mut fv = fx;

    // e is the distance moved on the step before last.
    // d is the most recent step.
    let mut e = 0.0_f64;
    let mut d = 0.0_f64;

    for _iter in 0..max_iter {
        let midpoint = 0.5 * (lo + hi);
        let tol1 = tol * x.abs() + 1e-10;
        let tol2 = 2.0 * tol1;

        // Convergence check: bracket is narrow enough.
        if (x - midpoint).abs() <= tol2 - 0.5 * (hi - lo) {
            return Minimize1DResult {
                x,
                fun: fx,
                nfev,
                success: true,
            };
        }

        // Try parabolic interpolation.
        let mut use_golden = true;

        if e.abs() > tol1 {
            // Fit a parabola through x, v, w.
            let r = (x - w) * (fx - fv);
            let q = (x - v) * (fx - fw);
            let mut p = (x - v) * q - (x - w) * r;
            let mut q = 2.0 * (q - r);

            if q > 0.0 {
                p = -p;
            } else {
                q = -q;
            }

            // Accept parabolic step if it is:
            // 1. Within the bracket [lo, hi]
            // 2. Smaller than half the previous step (ensures superlinear convergence)
            if p.abs() < (0.5 * q * e).abs() && p > q * (lo - x) && p < q * (hi - x) {
                d = p / q;
                let u = x + d;

                // Don't evaluate too close to the endpoints.
                if (u - lo) < tol2 || (hi - u) < tol2 {
                    d = if x < midpoint { tol1 } else { -tol1 };
                }
                use_golden = false;
            }
        }

        if use_golden {
            // Golden section step.
            e = if x < midpoint { hi - x } else { lo - x };
            d = golden * e;
        } else {
            e = d;
        }

        // Evaluate the function at the new point.
        let u = if d.abs() >= tol1 {
            x + d
        } else if d > 0.0 {
            x + tol1
        } else {
            x - tol1
        };

        let fu = f(u);
        nfev += 1;

        // Update the bracket and best points.
        if fu <= fx {
            if u < x {
                hi = x;
            } else {
                lo = x;
            }
            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        } else {
            if u < x {
                lo = u;
            } else {
                hi = u;
            }

            if fu <= fw || (w - x).abs() < 1e-30 {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv || (v - x).abs() < 1e-30 || (v - w).abs() < 1e-30 {
                v = u;
                fv = fu;
            }
        }
    }

    // Did not converge within max_iter, return best found.
    Minimize1DResult {
        x,
        fun: fx,
        nfev,
        success: false,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{Array1, array};

    // -----------------------------------------------------------------------
    // Helpers: Quadratic f(x) = 0.5 x^T A x - b^T x
    // -----------------------------------------------------------------------

    /// Returns (fun_grad, hessp) for a 3-dimensional quadratic with
    /// A = diag(2, 4, 6) and b = [1, 2, 3].
    ///
    /// The solution is x* = A^{-1} b = [0.5, 0.5, 0.5].
    #[allow(clippy::type_complexity)]
    fn quadratic_3d() -> (
        impl FnMut(&Array1<f64>) -> (f64, Array1<f64>),
        impl FnMut(&Array1<f64>, &Array1<f64>) -> Array1<f64>,
    ) {
        let a_diag = array![2.0, 4.0, 6.0];
        let b = array![1.0, 2.0, 3.0];

        let a_diag2 = a_diag.clone();
        let b2 = b.clone();

        let fun_grad = move |x: &Array1<f64>| {
            let ax = &a_diag * x;
            let f_val = 0.5 * x.dot(&ax) - x.dot(&b);
            let g = &ax - &b;
            (f_val, g)
        };

        let hessp = move |_x: &Array1<f64>, p: &Array1<f64>| &a_diag2 * p + &(&b2 * 0.0);

        (fun_grad, hessp)
    }

    // -----------------------------------------------------------------------
    // Helpers: Rosenbrock f(x,y) = (1-x)^2 + 100(y - x^2)^2
    // -----------------------------------------------------------------------

    fn rosenbrock_fun_grad(x: &Array1<f64>) -> (f64, Array1<f64>) {
        let a = 1.0 - x[0];
        let b = x[1] - x[0] * x[0];
        let f = a * a + 100.0 * b * b;
        let g = array![-2.0 * a - 400.0 * x[0] * b, 200.0 * b,];
        (f, g)
    }

    fn rosenbrock_hessp(x: &Array1<f64>, p: &Array1<f64>) -> Array1<f64> {
        // H = [[2 - 400(y - x^2) + 800 x^2,  -400 x],
        //      [-400 x,                        200   ]]
        let h00 = 2.0 - 400.0 * (x[1] - x[0] * x[0]) + 800.0 * x[0] * x[0];
        let h01 = -400.0 * x[0];
        let h11 = 200.0;
        array![h00 * p[0] + h01 * p[1], h01 * p[0] + h11 * p[1],]
    }

    // -----------------------------------------------------------------------
    // Newton-CG tests
    // -----------------------------------------------------------------------

    #[test]
    fn newton_cg_quadratic() {
        let (fun_grad, hessp) = quadratic_3d();
        let x0 = array![10.0, -5.0, 3.0];

        let result = NewtonCG::new()
            .minimize(fun_grad, hessp, x0)
            .expect("optimization should succeed");

        assert!(result.converged, "should converge on a quadratic");
        assert_abs_diff_eq!(result.x[0], 0.5, epsilon = 1e-8);
        assert_abs_diff_eq!(result.x[1], 0.5, epsilon = 1e-8);
        assert_abs_diff_eq!(result.x[2], 0.5, epsilon = 1e-8);
    }

    #[test]
    fn newton_cg_rosenbrock() {
        let x0 = array![-1.0, 1.0];

        let result = NewtonCG::new()
            .with_max_iter(500)
            .with_tol(1e-10)
            .minimize(rosenbrock_fun_grad, rosenbrock_hessp, x0)
            .expect("optimization should succeed");

        assert!(result.converged, "should converge on Rosenbrock");
        assert_abs_diff_eq!(result.x[0], 1.0, epsilon = 1e-4);
        assert_abs_diff_eq!(result.x[1], 1.0, epsilon = 1e-4);
    }

    // -----------------------------------------------------------------------
    // Trust-Region NCG tests
    // -----------------------------------------------------------------------

    #[test]
    fn trust_region_quadratic() {
        let (fun_grad, hessp) = quadratic_3d();
        let x0 = array![10.0, -5.0, 3.0];

        let result = TrustRegionNCG::new()
            .minimize(fun_grad, hessp, x0)
            .expect("optimization should succeed");

        assert!(result.converged, "should converge on a quadratic");
        assert_abs_diff_eq!(result.x[0], 0.5, epsilon = 1e-8);
        assert_abs_diff_eq!(result.x[1], 0.5, epsilon = 1e-8);
        assert_abs_diff_eq!(result.x[2], 0.5, epsilon = 1e-8);
    }

    #[test]
    fn trust_region_rosenbrock() {
        let x0 = array![-1.0, 1.0];

        let result = TrustRegionNCG::new()
            .with_max_iter(500)
            .with_tol(1e-10)
            .minimize(rosenbrock_fun_grad, rosenbrock_hessp, x0)
            .expect("optimization should succeed");

        assert!(result.converged, "should converge on Rosenbrock");
        assert_abs_diff_eq!(result.x[0], 1.0, epsilon = 1e-4);
        assert_abs_diff_eq!(result.x[1], 1.0, epsilon = 1e-4);
    }

    #[test]
    fn trust_region_high_dimensional() {
        // 10-dimensional quadratic: f(x) = 0.5 sum_i (i+1) x_i^2 - sum_i x_i
        // Solution: x_i = 1 / (i+1)
        let n = 10;
        let diag: Array1<f64> = (1..=n).map(|i| i as f64).collect();
        let b = Array1::ones(n);

        let diag2 = diag.clone();

        let fun_grad = move |x: &Array1<f64>| {
            let ax = &diag * x;
            let f_val = 0.5 * x.dot(&ax) - x.dot(&b);
            let g = &ax - &b;
            (f_val, g)
        };

        let hessp = move |_x: &Array1<f64>, p: &Array1<f64>| &diag2 * p;

        let x0 = Array1::from_elem(n, 5.0);

        let result = TrustRegionNCG::new()
            .minimize(fun_grad, hessp, x0)
            .expect("optimization should succeed");

        assert!(result.converged, "should converge on 10-d quadratic");
        for i in 0..n {
            let expected = 1.0 / (i + 1) as f64;
            assert_abs_diff_eq!(result.x[i], expected, epsilon = 1e-8);
        }
    }

    #[test]
    fn newton_cg_convergence_flag() {
        // Easy problem: should converge with default settings.
        let (fun_grad, hessp) = quadratic_3d();
        let x0 = array![1.0, 1.0, 1.0];

        let result = NewtonCG::new()
            .minimize(fun_grad, hessp, x0)
            .expect("optimization should succeed");
        assert!(result.converged, "should converge on easy problem");

        // Same problem but max_iter=1: should NOT converge.
        let (fun_grad2, hessp2) = quadratic_3d();
        let x0 = array![10.0, -5.0, 3.0];

        let result2 = NewtonCG::new()
            .with_max_iter(1)
            .minimize(fun_grad2, hessp2, x0)
            .expect("optimization should succeed (even if not converged)");
        assert!(
            !result2.converged,
            "should not converge with only 1 iteration from a distant start"
        );
    }

    // -----------------------------------------------------------------------
    // Brent bounded 1-D minimization tests
    // -----------------------------------------------------------------------

    #[test]
    fn brent_minimize_x_squared() {
        // Minimize x^2 on [-1, 2] → minimum at x = 0.
        let result = super::brent_bounded(|x| x * x, -1.0, 2.0, 1e-10, 500);
        assert!(result.success, "should converge");
        assert_abs_diff_eq!(result.x, 0.0, epsilon = 1e-8);
        assert_abs_diff_eq!(result.fun, 0.0, epsilon = 1e-15);
    }

    #[test]
    fn brent_minimize_shifted_quadratic() {
        // Minimize (x - 1.5)^2 on [0, 3] → minimum at x = 1.5.
        let result = super::brent_bounded(|x| (x - 1.5) * (x - 1.5), 0.0, 3.0, 1e-10, 500);
        assert!(result.success, "should converge");
        assert_abs_diff_eq!(result.x, 1.5, epsilon = 1e-8);
        assert_abs_diff_eq!(result.fun, 0.0, epsilon = 1e-15);
    }

    #[test]
    fn brent_minimize_sin() {
        // Minimize sin(x) on [3, 6] → minimum at x = 3π/2 ≈ 4.71239.
        let result = super::brent_bounded(f64::sin, 3.0, 6.0, 1e-10, 500);
        assert!(result.success, "should converge");
        let expected = 1.5 * std::f64::consts::PI;
        assert_abs_diff_eq!(result.x, expected, epsilon = 1e-8);
        assert_abs_diff_eq!(result.fun, -1.0, epsilon = 1e-10);
    }

    // -----------------------------------------------------------------------
    // Powell direction-set tests
    // -----------------------------------------------------------------------

    #[test]
    fn powell_finds_2d_quadratic_minimum() {
        // f(x, y) = (x - 3)^2 + (y + 4)^2 → min at (3, -4), f = 0.
        let f = |x: &Array1<f64>| (x[0] - 3.0).powi(2) + (x[1] + 4.0).powi(2);
        let r = super::Powell::new().minimize(f, array![0.0, 0.0]);
        assert!(r.converged, "did not converge in {} iter", r.n_iter);
        assert_abs_diff_eq!(r.x[0], 3.0, epsilon = 1e-4);
        assert_abs_diff_eq!(r.x[1], -4.0, epsilon = 1e-4);
        assert!(r.fun < 1e-8, "f = {}", r.fun);
        // Derivative-free: grad is reported as zeros.
        assert_eq!(r.grad.len(), 2);
        assert!(r.grad.iter().all(|&g| g == 0.0));
    }

    #[test]
    fn powell_finds_skewed_quadratic() {
        // f(x, y) = (x - y)^2 + (x + y - 1)^2 → min at (0.5, 0.5).
        let f = |x: &Array1<f64>| (x[0] - x[1]).powi(2) + (x[0] + x[1] - 1.0).powi(2);
        let r = super::Powell::new().minimize(f, array![3.0, -2.0]);
        assert_abs_diff_eq!(r.x[0], 0.5, epsilon = 1e-4);
        assert_abs_diff_eq!(r.x[1], 0.5, epsilon = 1e-4);
    }

    #[test]
    fn powell_3d_anisotropic_quadratic() {
        // f(dy, dx, theta) = (dy - 0.7)^2 + (dx + 1.2)^2 + 100*(theta - 0.04)^2.
        // The 100× factor on theta is realistic for motion-correction
        // objectives where rotation has much sharper curvature than
        // translation — Powell handles the anisotropy via the direction
        // updates.
        let f = |x: &Array1<f64>| {
            (x[0] - 0.7).powi(2) + (x[1] + 1.2).powi(2) + 100.0 * (x[2] - 0.04).powi(2)
        };
        let r = super::Powell::new()
            .with_initial_step(0.2)
            .minimize(f, array![0.0, 0.0, 0.0]);
        assert_abs_diff_eq!(r.x[0], 0.7, epsilon = 1e-3);
        assert_abs_diff_eq!(r.x[1], -1.2, epsilon = 1e-3);
        assert_abs_diff_eq!(r.x[2], 0.04, epsilon = 1e-4);
    }

    #[test]
    fn powell_iteration_limit_marks_unconverged() {
        // A pathologically tight max_iter on a curved 5-D objective should
        // hit the iteration limit and report converged=false.
        let f = |x: &Array1<f64>| (0..x.len()).map(|i| (x[i] - i as f64).powi(2)).sum();
        let r = super::Powell::new()
            .with_max_iter(1)
            .minimize(f, array![10.0, 10.0, 10.0, 10.0, 10.0]);
        assert!(!r.converged, "1-iter run should not declare convergence");
        assert_eq!(r.n_iter, 1);
    }

    #[test]
    fn powell_one_dim_reduces_to_line_search() {
        // 1-D problem: f(x) = (x - 7)^2 + 1 → min at x = 7, f = 1.
        let f = |x: &Array1<f64>| (x[0] - 7.0).powi(2) + 1.0;
        let r = super::Powell::new().minimize(f, array![0.0]);
        assert!(r.converged);
        assert_abs_diff_eq!(r.x[0], 7.0, epsilon = 1e-4);
        assert_abs_diff_eq!(r.fun, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn powell_handles_fnmut_closure_with_eval_counter() {
        // FnMut bound is real: the closure can mutably borrow external
        // state. Counting evaluations is the canonical use case.
        let mut nfev = 0_usize;
        let r = super::Powell::new().minimize(
            |x: &Array1<f64>| {
                nfev += 1;
                (x[0] - 1.5).powi(2) + (x[1] + 0.25).powi(2)
            },
            array![0.0, 0.0],
        );
        assert!(r.converged);
        assert_abs_diff_eq!(r.x[0], 1.5, epsilon = 1e-4);
        assert_abs_diff_eq!(r.x[1], -0.25, epsilon = 1e-4);
        assert!(nfev > 0, "closure should have been called");
    }
}
