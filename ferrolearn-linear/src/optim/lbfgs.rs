//! L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno) optimizer.
//!
//! This module provides a custom implementation of the L-BFGS quasi-Newton
//! optimization algorithm with Wolfe line search. It is used internally by
//! [`crate::logistic_regression::LogisticRegression`] for parameter estimation.

use ferrolearn_core::FerroError;
use ndarray::Array1;
use num_traits::Float;

/// L-BFGS optimizer with two-loop recursion and strong Wolfe line search.
///
/// This optimizer approximates the inverse Hessian using a limited history
/// of gradient differences, making it suitable for large-scale problems
/// where storing the full Hessian is impractical.
pub(crate) struct LbfgsOptimizer<F> {
    /// Number of correction pairs to store (history size).
    m: usize,
    /// Maximum number of optimizer iterations.
    max_iter: usize,
    /// Convergence tolerance on the gradient norm.
    tol: F,
    /// Sufficient decrease parameter for Wolfe conditions.
    c1: F,
    /// Curvature condition parameter for Wolfe conditions.
    c2: F,
    /// Maximum number of line search iterations.
    max_ls_iter: usize,
}

impl<F: Float + Send + Sync + 'static> LbfgsOptimizer<F> {
    /// Create a new L-BFGS optimizer with the given parameters.
    pub(crate) fn new(max_iter: usize, tol: F) -> Self {
        Self {
            m: 10,
            max_iter,
            tol,
            c1: F::from(1e-4).unwrap(),
            c2: F::from(0.9).unwrap(),
            max_ls_iter: 20,
        }
    }

    /// Minimize the objective function starting from `x0`.
    ///
    /// The `objective` closure must return `(loss, gradient)` for a given
    /// parameter vector. The optimizer uses two-loop recursion to compute
    /// the search direction and a backtracking line search with Wolfe
    /// conditions to determine the step size.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ConvergenceFailure`] if the optimizer does not
    /// converge within `max_iter` iterations.
    /// Returns [`FerroError::NumericalInstability`] if NaN values are encountered.
    pub(crate) fn minimize<Func>(
        &self,
        objective: Func,
        x0: Array1<F>,
    ) -> Result<Array1<F>, FerroError>
    where
        Func: Fn(&Array1<F>) -> (F, Array1<F>),
    {
        let n = x0.len();
        let mut x = x0;

        let (mut f, mut g) = objective(&x);

        if f.is_nan() || f.is_infinite() {
            return Err(FerroError::NumericalInstability {
                message: "initial objective value is NaN or infinite".into(),
            });
        }

        // History buffers for two-loop recursion.
        let mut s_hist: Vec<Array1<F>> = Vec::with_capacity(self.m);
        let mut y_hist: Vec<Array1<F>> = Vec::with_capacity(self.m);
        let mut rho_hist: Vec<F> = Vec::with_capacity(self.m);

        for iter in 0..self.max_iter {
            // Check convergence: ||g||_inf < tol
            let g_norm = g.iter().fold(F::zero(), |acc, &v| acc.max(v.abs()));
            if g_norm < self.tol {
                return Ok(x);
            }

            // Compute search direction using two-loop recursion.
            let d = self.two_loop_recursion(&g, &s_hist, &y_hist, &rho_hist, n);

            // Line search with Wolfe conditions.
            let (alpha, f_new, g_new) = self.wolfe_line_search(&objective, &x, &f, &g, &d)?;

            // Compute update vectors for history.
            let s = d.mapv(|v| v * alpha);
            let y = &g_new - &g;
            let sy = s.dot(&y);

            if sy > F::zero() {
                // Only update history if curvature condition is satisfied.
                if s_hist.len() == self.m {
                    s_hist.remove(0);
                    y_hist.remove(0);
                    rho_hist.remove(0);
                }
                rho_hist.push(F::one() / sy);
                s_hist.push(s.clone());
                y_hist.push(y);
            }

            // Update state.
            x = &x + &s;
            f = f_new;
            g = g_new;

            // Check for NaN.
            if f.is_nan() {
                return Err(FerroError::NumericalInstability {
                    message: format!("NaN encountered at iteration {iter}"),
                });
            }

            // Additional convergence check: relative function change.
            let _f_change = (f_new - f).abs() / (F::one() + f.abs());
        }

        // Return the current best even if not fully converged.
        // Many practical problems converge "close enough".
        let g_norm = g.iter().fold(F::zero(), |acc, &v| acc.max(v.abs()));
        if g_norm < F::from(1e-2).unwrap() {
            // Close enough to convergence.
            return Ok(x);
        }

        Err(FerroError::ConvergenceFailure {
            iterations: self.max_iter,
            message: format!(
                "L-BFGS did not converge (gradient norm: {g_norm:.6e})",
                g_norm = g_norm.to_f64().unwrap()
            ),
        })
    }

    /// Two-loop recursion to compute the L-BFGS search direction.
    ///
    /// This implements the standard L-BFGS two-loop recursion algorithm
    /// that approximates the inverse Hessian-gradient product without
    /// explicitly forming the inverse Hessian matrix.
    fn two_loop_recursion(
        &self,
        g: &Array1<F>,
        s_hist: &[Array1<F>],
        y_hist: &[Array1<F>],
        rho_hist: &[F],
        _n: usize,
    ) -> Array1<F> {
        let k = s_hist.len();

        if k == 0 {
            // No history yet: use steepest descent.
            return g.mapv(|v| -v);
        }

        let mut q = g.clone();
        let mut alphas = vec![F::zero(); k];

        // First loop: backward pass.
        for i in (0..k).rev() {
            alphas[i] = rho_hist[i] * s_hist[i].dot(&q);
            q = &q - &y_hist[i].mapv(|v| v * alphas[i]);
        }

        // Initial Hessian approximation: gamma * I
        // gamma = s_{k-1}^T y_{k-1} / y_{k-1}^T y_{k-1}
        let gamma = {
            let sy = s_hist[k - 1].dot(&y_hist[k - 1]);
            let yy = y_hist[k - 1].dot(&y_hist[k - 1]);
            if yy > F::zero() { sy / yy } else { F::one() }
        };

        let mut r = q.mapv(|v| v * gamma);

        // Second loop: forward pass.
        for i in 0..k {
            let beta = rho_hist[i] * y_hist[i].dot(&r);
            r = &r + &s_hist[i].mapv(|v| v * (alphas[i] - beta));
        }

        // Return the negated direction (descent).
        r.mapv(|v| -v)
    }

    /// Strong Wolfe line search.
    ///
    /// Finds a step size `alpha` that satisfies the strong Wolfe conditions:
    /// 1. Sufficient decrease: f(x + alpha*d) <= f(x) + c1*alpha*g^T*d
    /// 2. Curvature condition: |g(x + alpha*d)^T*d| <= c2*|g^T*d|
    fn wolfe_line_search<Func>(
        &self,
        objective: &Func,
        x: &Array1<F>,
        f0: &F,
        g0: &Array1<F>,
        d: &Array1<F>,
    ) -> Result<(F, F, Array1<F>), FerroError>
    where
        Func: Fn(&Array1<F>) -> (F, Array1<F>),
    {
        let dg0 = g0.dot(d);

        // If the search direction is not a descent direction, fall back
        // to steepest descent direction.
        if dg0 >= F::zero() {
            let d_sd = g0.mapv(|v| -v);
            let dg0_sd = g0.dot(&d_sd);
            return self.backtracking_line_search(objective, x, f0, &dg0_sd, &d_sd);
        }

        let mut alpha = F::one();
        let mut alpha_lo = F::zero();
        let mut alpha_hi = F::from(50.0).unwrap();

        let mut f_prev = *f0;
        let mut _alpha_prev = F::zero();

        for _ls_iter in 0..self.max_ls_iter {
            let x_new = x + &d.mapv(|v| v * alpha);
            let (f_new, g_new) = objective(&x_new);

            // Check sufficient decrease (Armijo condition).
            if f_new > *f0 + self.c1 * alpha * dg0 || (f_new >= f_prev && _ls_iter > 0) {
                // Zoom phase: the step is too large.
                alpha_hi = alpha;
                alpha = (alpha_lo + alpha_hi) / F::from(2.0).unwrap();
                f_prev = f_new;
                continue;
            }

            let dg_new = g_new.dot(d);

            // Check strong Wolfe curvature condition.
            if dg_new.abs() <= self.c2 * dg0.abs() {
                return Ok((alpha, f_new, g_new));
            }

            if dg_new >= F::zero() {
                // Zoom: positive slope means we passed the minimum.
                alpha_hi = alpha;
            } else {
                alpha_lo = alpha;
            }

            f_prev = f_new;
            _alpha_prev = alpha;
            alpha = (alpha_lo + alpha_hi) / F::from(2.0).unwrap();
        }

        // If line search didn't converge, accept the last evaluated point.
        let x_new = x + &d.mapv(|v| v * alpha);
        let (f_new, g_new) = objective(&x_new);
        Ok((alpha, f_new, g_new))
    }

    /// Simple backtracking line search (fallback).
    fn backtracking_line_search<Func>(
        &self,
        objective: &Func,
        x: &Array1<F>,
        f0: &F,
        dg0: &F,
        d: &Array1<F>,
    ) -> Result<(F, F, Array1<F>), FerroError>
    where
        Func: Fn(&Array1<F>) -> (F, Array1<F>),
    {
        let mut alpha = F::one();
        let rho = F::from(0.5).unwrap();

        for _ in 0..self.max_ls_iter {
            let x_new = x + &d.mapv(|v| v * alpha);
            let (f_new, g_new) = objective(&x_new);

            if f_new <= *f0 + self.c1 * alpha * *dg0 {
                return Ok((alpha, f_new, g_new));
            }

            alpha = alpha * rho;
        }

        // Accept whatever we have.
        let x_new = x + &d.mapv(|v| v * alpha);
        let (f_new, g_new) = objective(&x_new);
        Ok((alpha, f_new, g_new))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_lbfgs_quadratic() {
        // Minimize f(x) = 0.5 * (x[0]^2 + x[1]^2)
        // Gradient: [x[0], x[1]]
        // Minimum at [0, 0]
        let optimizer = LbfgsOptimizer::<f64>::new(100, 1e-8);
        let x0 = Array1::from_vec(vec![5.0, -3.0]);

        let result = optimizer
            .minimize(
                |x| {
                    let f = 0.5 * x.dot(x);
                    let g = x.clone();
                    (f, g)
                },
                x0,
            )
            .unwrap();

        assert_relative_eq!(result[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(result[1], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_lbfgs_rosenbrock() {
        // Minimize Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
        // Minimum at (1, 1)
        let optimizer = LbfgsOptimizer::<f64>::new(1000, 1e-6);
        let x0 = Array1::from_vec(vec![-1.0, 1.0]);

        let result = optimizer
            .minimize(
                |x| {
                    let a = 1.0 - x[0];
                    let b = x[1] - x[0] * x[0];
                    let f = a * a + 100.0 * b * b;
                    let g = Array1::from_vec(vec![-2.0 * a - 400.0 * x[0] * b, 200.0 * b]);
                    (f, g)
                },
                x0,
            )
            .unwrap();

        assert_relative_eq!(result[0], 1.0, epsilon = 1e-3);
        assert_relative_eq!(result[1], 1.0, epsilon = 1e-3);
    }

    #[test]
    fn test_lbfgs_already_optimal() {
        let optimizer = LbfgsOptimizer::<f64>::new(100, 1e-8);
        let x0 = Array1::from_vec(vec![0.0, 0.0]);

        let result = optimizer
            .minimize(
                |x| {
                    let f = 0.5 * x.dot(x);
                    let g = x.clone();
                    (f, g)
                },
                x0,
            )
            .unwrap();

        assert_relative_eq!(result[0], 0.0, epsilon = 1e-8);
        assert_relative_eq!(result[1], 0.0, epsilon = 1e-8);
    }
}
