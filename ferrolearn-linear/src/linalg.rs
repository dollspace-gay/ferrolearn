//! Internal linear algebra utilities.
//!
//! This module provides helper functions for solving linear systems. The
//! unregularized least-squares (OLS) path mirrors scikit-learn's dense solve
//! `self.coef_, _, self.rank_, self.singular_ = linalg.lstsq(X, y)`
//! (`sklearn/linear_model/_base.py:687`) — `scipy.linalg.lstsq` → LAPACK
//! `gelsd` (SVD-based, minimum-norm) — by routing through
//! [`ferray::linalg::lstsq`](ferray::linalg::lstsq) (`ferray-linalg/src/solve.rs:208`),
//! a single-SVD gelsd-equivalent solver that zeroes sub-`rcond` singular
//! values (yielding the minimum-norm solution) and accepts any `m × n` system
//! (underdetermined included). This is the ferray substrate (R-SUBSTRATE-1).
//! The Ridge path retains its hand-rolled Cholesky kernels (positive-definite
//! for `alpha > 0`, where the min-norm concern does not arise).
//!
//! The `ndarray ↔ ferray` conversion happens at this module boundary
//! (R-SUBSTRATE-4): callers keep their `ndarray` signatures during the
//! workspace-wide migration.

use ferray::linalg::LinalgFloat;
use ferray::{Array as FerrayArray, IxDyn};
use ferrolearn_core::FerroError;
use ndarray::{Array1, Array2};
use num_traits::Float;

/// Solve the least squares problem `X @ w = y` for `w`.
///
/// Routes through [`ferray::linalg::lstsq`] (`ferray-linalg/src/solve.rs:208`),
/// a single-SVD, LAPACK-`gelsd`-equivalent solver. For a rank-deficient or
/// underdetermined `X` it returns the unique **minimum-norm** least-squares
/// solution (sub-`rcond` singular values are zeroed), matching scikit-learn's
/// `linalg.lstsq(X, y)` (`sklearn/linear_model/_base.py:687`). `rcond` is left
/// at the `None` default (`max(m, n) * eps`), matching scipy/sklearn's default.
///
/// Any `m × n` shape is accepted, including `n_samples < n_features`
/// (underdetermined), exactly as `linalg.lstsq` does.
///
/// # Errors
///
/// Returns [`FerroError::NumericalInstability`] if the underlying SVD fails
/// or the ferray↔ndarray bridge encounters a shape inconsistency.
pub(crate) fn solve_lstsq<F: LinalgFloat>(
    x: &Array2<F>,
    y: &Array1<F>,
) -> Result<Array1<F>, FerroError> {
    let (n_samples, n_features) = x.dim();

    // Bridge ndarray -> ferray (R-SUBSTRATE-4). Build from a flat,
    // row-major Vec + shape; ferray-core's `from_ndarray` is crate-private.
    let x_flat: Vec<F> = x.iter().copied().collect();
    let a =
        FerrayArray::<F, ferray::Ix2>::from_vec(ferray::Ix2::new([n_samples, n_features]), x_flat)
            .map_err(|e| FerroError::NumericalInstability {
                message: format!("ferray lstsq: failed to build design matrix: {e}"),
            })?;

    let y_flat: Vec<F> = y.iter().copied().collect();
    let b = FerrayArray::<F, IxDyn>::from_vec(IxDyn::new(&[n_samples]), y_flat).map_err(|e| {
        FerroError::NumericalInstability {
            message: format!("ferray lstsq: failed to build target vector: {e}"),
        }
    })?;

    // Single-SVD gelsd-equivalent solve; `None` rcond matches the
    // scipy/sklearn default of `max(m, n) * eps`.
    let (solution, _residuals, _rank, _singular) =
        ferray::linalg::lstsq(&a, &b, None).map_err(|e| FerroError::NumericalInstability {
            message: format!("ferray lstsq solve failed: {e}"),
        })?;

    // Bridge ferray -> ndarray: solution is a 1-D `IxDyn` array of length
    // `n_features`. `into_ndarray()` yields an `ndarray::ArrayD`; flatten to
    // the owned `Array1<F>` callers expect.
    let solution_nd = solution.into_ndarray();
    let w_vec: Vec<F> = solution_nd.iter().copied().collect();
    if w_vec.len() != n_features {
        return Err(FerroError::NumericalInstability {
            message: format!(
                "ferray lstsq: solution length {} does not match {} features",
                w_vec.len(),
                n_features
            ),
        });
    }
    Ok(Array1::from_vec(w_vec))
}

/// Solve a symmetric positive-definite system `A @ x = b` via Cholesky.
fn cholesky_solve<F: Float>(a: &Array2<F>, b: &Array1<F>) -> Result<Array1<F>, FerroError> {
    let n = a.nrows();

    // Compute lower triangular L such that A = L @ L^T.
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
                        message: "matrix is not positive definite".into(),
                    });
                }
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }

    // Forward substitution: L @ z = b
    let mut z = Array1::<F>::zeros(n);
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum = sum - l[[i, j]] * z[j];
        }
        z[i] = sum / l[[i, i]];
    }

    // Backward substitution: L^T @ x = z
    let mut x = Array1::<F>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = z[i];
        for j in (i + 1)..n {
            sum = sum - l[[j, i]] * x[j];
        }
        x[i] = sum / l[[i, i]];
    }

    Ok(x)
}

/// Solve `A @ x = b` via Gaussian elimination with partial pivoting.
fn gaussian_solve<F: Float>(
    n: usize,
    a: &Array2<F>,
    b: &Array1<F>,
) -> Result<Array1<F>, FerroError> {
    // Augmented matrix [A | b].
    let mut aug = Array2::<F>::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    // Forward elimination with partial pivoting.
    for col in 0..n {
        // Find pivot.
        let mut max_val = aug[[col, col]].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let val = aug[[row, col]].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_val < F::from(1e-12).unwrap_or_else(F::epsilon) {
            return Err(FerroError::NumericalInstability {
                message: "singular matrix encountered during Gaussian elimination".into(),
            });
        }

        // Swap rows.
        if max_row != col {
            for j in 0..=n {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }
        }

        // Eliminate below.
        let pivot = aug[[col, col]];
        for row in (col + 1)..n {
            let factor = aug[[row, col]] / pivot;
            for j in col..=n {
                let above = aug[[col, j]];
                aug[[row, j]] = aug[[row, j]] - factor * above;
            }
        }
    }

    // Back substitution.
    let mut x = Array1::<F>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = aug[[i, n]];
        for j in (i + 1)..n {
            sum = sum - aug[[i, j]] * x[j];
        }
        if aug[[i, i]].abs() < F::from(1e-12).unwrap_or_else(F::epsilon) {
            return Err(FerroError::NumericalInstability {
                message: "near-zero pivot during back substitution".into(),
            });
        }
        x[i] = sum / aug[[i, i]];
    }

    Ok(x)
}

/// Solve a symmetric positive-definite system `A @ X = B` via Cholesky,
/// where `B` is `(n, t)` and the returned `X` is `(n, t)`. Each column of
/// `B` is solved independently after a single Cholesky factorization of
/// `A` — the asymptotic win vs. calling [`cholesky_solve`] in a loop is
/// the factorization cost `O(n^3)` paid once instead of `t` times.
///
/// Used by [`solve_ridge_multi`] to share `X^T X + alpha * I`'s
/// factorization across all targets.
fn cholesky_solve_multi<F: Float>(a: &Array2<F>, b: &Array2<F>) -> Result<Array2<F>, FerroError> {
    let n = a.nrows();
    let t = b.ncols();

    // Cholesky-Crout: A = L @ L^T, L lower-triangular.
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
                        message: "matrix is not positive definite".into(),
                    });
                }
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }

    // For each target column independently: forward then backward sub.
    let mut out = Array2::<F>::zeros((n, t));
    for k in 0..t {
        // Forward sub: L @ z = b[:, k]
        let mut z = Array1::<F>::zeros(n);
        for i in 0..n {
            let mut sum = b[[i, k]];
            for j in 0..i {
                sum = sum - l[[i, j]] * z[j];
            }
            z[i] = sum / l[[i, i]];
        }
        // Backward sub: L^T @ x = z, write into out[:, k]
        for i in (0..n).rev() {
            let mut sum = z[i];
            for j in (i + 1)..n {
                sum = sum - l[[j, i]] * out[[j, k]];
            }
            out[[i, k]] = sum / l[[i, i]];
        }
    }

    Ok(out)
}

/// Solve `(X^T X + alpha * I) @ w = X^T y` (Ridge regression).
///
/// Uses Cholesky decomposition since `X^T X + alpha * I` is guaranteed
/// to be positive definite for `alpha > 0`.
///
/// # Errors
///
/// Returns [`FerroError::NumericalInstability`] if the regularized system
/// is somehow singular (should not happen for `alpha > 0`).
pub(crate) fn solve_ridge<F: Float + Send + Sync + 'static>(
    x: &Array2<F>,
    y: &Array1<F>,
    alpha: F,
) -> Result<Array1<F>, FerroError> {
    let xt = x.t();
    let mut xtx = xt.dot(x);
    let xty = xt.dot(y);
    let n = xtx.nrows();

    // Add regularization: X^T X + alpha * I
    for i in 0..n {
        xtx[[i, i]] = xtx[[i, i]] + alpha;
    }

    cholesky_solve(&xtx, &xty).or_else(|_| gaussian_solve(n, &xtx, &xty))
}

/// Solve `(X^T X + alpha * I) @ W = X^T Y` (multi-output Ridge regression).
///
/// `X` is `(n_samples, n_features)`, `Y` is `(n_samples, n_targets)`, and
/// the returned `W` is `(n_features, n_targets)`. The Cholesky factor of
/// `X^T X + alpha * I` is shared across all target columns, so the cost
/// is dominated by one `O(p^3)` factorization plus `O(p^2 * t)` for the
/// forward/backward substitutions — the same asymptotic behaviour as a
/// single-output fit on `t = 1`. This is the multi-output companion to
/// [`solve_ridge`].
///
/// # Errors
///
/// Returns [`FerroError::NumericalInstability`] if the regularized system
/// is somehow singular (should not happen for `alpha > 0`).
pub(crate) fn solve_ridge_multi<F: Float + Send + Sync + 'static>(
    x: &Array2<F>,
    y: &Array2<F>,
    alpha: F,
) -> Result<Array2<F>, FerroError> {
    let xt = x.t();
    let mut xtx = xt.dot(x);
    let xty = xt.dot(y);
    let n = xtx.nrows();

    // Add regularization: X^T X + alpha * I
    for i in 0..n {
        xtx[[i, i]] = xtx[[i, i]] + alpha;
    }

    cholesky_solve_multi(&xtx, &xty)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_solve_lstsq_simple() {
        // 2x = 4 -> x = 2
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = Array1::from_vec(vec![2.0, 4.0, 6.0]);
        let w = solve_lstsq(&x, &y).unwrap();
        assert_relative_eq!(w[0], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_solve_lstsq_multi() {
        // y = x1 + 2*x2
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let w = solve_lstsq(&x, &y).unwrap();
        assert_relative_eq!(w[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(w[1], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_solve_ridge() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = Array1::from_vec(vec![2.0, 4.0, 6.0]);
        let w = solve_ridge(&x, &y, 0.0).unwrap();
        assert_relative_eq!(w[0], 2.0, epsilon = 1e-10);

        // With regularization, coefficients should shrink.
        let w_reg = solve_ridge(&x, &y, 10.0).unwrap();
        assert!(w_reg[0].abs() < w[0].abs());
    }

    #[test]
    fn test_solve_lstsq_rank_deficient_min_norm() {
        // Rank-1 design (duplicate columns). The minimum-norm least-squares
        // solution splits the weight evenly across the tied columns. Oracle:
        //   python3 -c "import numpy as np; from scipy.linalg import lstsq; \
        //     print(lstsq(np.array([[1.,1.],[2.,2.],[3.,3.]]), \
        //     np.array([1.,2.,3.]))[0].tolist())"  -> [0.5, 0.5]
        // (the gelsd min-norm split; the same value sklearn
        // LinearRegression(fit_intercept=False) returns, per
        // tests/divergence_linreg_minnorm.rs).
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]).unwrap();
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let w = solve_lstsq(&x, &y).unwrap();
        assert_relative_eq!(w[0], 0.5, epsilon = 1e-10);
        assert_relative_eq!(w[1], 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_solve_lstsq_underdetermined_accepted() {
        // n_samples (2) < n_features (3): scipy.linalg.lstsq accepts this and
        // returns the minimum-norm solution. Oracle:
        //   python3 -c "import numpy as np; from scipy.linalg import lstsq; \
        //     print(lstsq(np.array([[1.,2.,3.],[4.,5.,6.]]), \
        //     np.array([1.,2.]))[0].tolist())"
        //   -> [-0.05555555555555583, 0.11111111111111112, 0.277777777777778]
        let x = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = Array1::from_vec(vec![1.0, 2.0]);
        let w = solve_lstsq(&x, &y).unwrap();
        assert_relative_eq!(w[0], -0.055_555_555_555_555_83, epsilon = 1e-8);
        assert_relative_eq!(w[1], 0.111_111_111_111_111_12, epsilon = 1e-8);
        assert_relative_eq!(w[2], 0.277_777_777_777_778, epsilon = 1e-8);
    }
}
