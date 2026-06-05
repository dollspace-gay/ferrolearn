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
//!
//! ## REQ status (per `.design/linear/linalg.md`, mirrors `sklearn/linear_model/_base.py:687` @ 1.5.2)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (full-rank OLS solve) | SHIPPED | `solve_lstsq` → `ferray::linalg::lstsq`; full-rank coef/intercept match the live sklearn oracle to 1e-8. Consumer: `Fit for LinearRegression in linear_regression.rs`. |
//! | REQ-2 (minimum-norm for rank-deficient X) | SHIPPED | `solve_lstsq` → `ferray::linalg::lstsq` (SVD zeroes sub-rcond sv → min-norm), mirrors LAPACK `gelsd` (`_base.py:687`). Closed #376; regression test `divergence_rank_deficient_*_min_norm`. |
//! | REQ-3 (underdetermined n<p accepted) | SHIPPED | rejection removed; SVD handles any m×n. Closed #377; regression test `divergence_underdetermined_accepted_min_norm`. |
//! | REQ-4 (rank_ / singular_ exposed) | NOT-STARTED | blocker #374 — `ferray::linalg::lstsq` returns rank+singular values, but they are not yet stored as fitted attributes on the estimators. |
//! | REQ-5 (safe_sparse_dot helper) | NOT-STARTED | blocker #380 — no dot/matmul wrapper (`extmath.py:161`); estimators call `ndarray::.dot()` inline. |
//! | REQ-6 (ferray substrate for OLS solve) | SHIPPED | OLS decomposition runs on `ferray::linalg` (`solve.rs:208`); ndarray↔ferray bridged at this boundary (R-SUBSTRATE-4). |
//! | REQ-7 (gelsd rcond cutoff parity, `eps * s_max`) | SHIPPED | `solve_lstsq` passes `Some(F::epsilon())` to `ferray::linalg::lstsq`, pinning the singular-value cutoff to scipy's `cond=eps` (matching `linalg.lstsq(X, y)`'s default, `_base.py:687`) so the RANK decision matches scipy/sklearn. Closed #381; regression test `lstsq_rcond_eps_cutoff_and_stable_contract`. Per #382 the residual coefficient values on a numerically-singular (`cond~1e14`) design are an inherent FP limit (1/s_min-amplified noise, no implementation has a "true" answer), so the deterministic contract asserted is rank parity + the stable `X @ coef` projection, not the individual coefficients. |
//!
//! acto-critic: #376/#377 fixed and verified vs the live oracle; full-rank parity,
//! bridge fidelity, and edge cases (single feature/sample, f32, fit_intercept) all
//! match. #381 (rcond cutoff) fixed: the `eps * s_max` cutoff makes the rank
//! decision match scipy/sklearn; per ferray #382 the coefficient values on a
//! numerically-singular design are an inherent FP limit, so the deterministic
//! contract is rank parity + the stable `X @ coef` projection.
//! Two states only per goal.md R-DEFER-2.
//!
//! The Ridge path retains its hand-rolled Cholesky kernels (PD for `alpha > 0`).

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
/// `linalg.lstsq(X, y)` (`sklearn/linear_model/_base.py:687`). `rcond` is set
/// to `Some(F::epsilon())` (machine eps), matching scipy's `cond=eps` cutoff
/// (`eps * s_max`) — the default scipy/sklearn use when `cond=None` — so the
/// singular-value rank decision matches scipy/sklearn (rather than ferray's
/// larger numpy-convention `max(m, n) * eps` default).
///
/// Any `m × n` shape is accepted, including `n_samples < n_features`
/// (underdetermined), exactly as `linalg.lstsq` does.
///
/// Returns `(solution, rank, singular_values)`: the minimum-norm
/// least-squares solution, the effective rank of `X` (sklearn `rank_`), and
/// the singular values of `X` (sklearn `singular_`), exactly the values
/// sklearn captures via `self.coef_, _, self.rank_, self.singular_ =
/// linalg.lstsq(X, y)` (`sklearn/linear_model/_base.py:687`).
///
/// # Errors
///
/// Returns [`FerroError::NumericalInstability`] if the underlying SVD fails
/// or the ferray↔ndarray bridge encounters a shape inconsistency.
pub(crate) fn solve_lstsq<F: LinalgFloat>(
    x: &Array2<F>,
    y: &Array1<F>,
) -> Result<(Array1<F>, usize, Array1<F>), FerroError> {
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

    // Single-SVD gelsd-equivalent solve. scikit-learn calls
    // `linalg.lstsq(X, y)` with no `cond` (`sklearn/linear_model/_base.py:687`);
    // scipy's `cond=None` default sets the singular-value cutoff to
    // `eps * s_max` (machine epsilon). ferray's own `None` default uses the
    // larger numpy convention `max(m, n) * eps * s_max`, which makes a
    // DIFFERENT rank decision for singular-value ratios in `(eps, max(m,n)*eps)`.
    // Passing `Some(F::epsilon())` pins ferray to scipy's `cond=eps` cutoff so
    // the rank decision matches scipy/sklearn.
    let (solution, _residuals, rank, singular) = ferray::linalg::lstsq(&a, &b, Some(F::epsilon()))
        .map_err(|e| FerroError::NumericalInstability {
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

    // Bridge the singular values (sklearn `singular_`) ferray -> ndarray, the
    // same flat-collect pattern as the solution. `rank` (sklearn `rank_`) is a
    // plain `usize`. These are exactly the values sklearn captures from
    // `linalg.lstsq(X, y)` (`sklearn/linear_model/_base.py:687`).
    let singular_nd = singular.into_ndarray();
    let singular_vec: Vec<F> = singular_nd.iter().copied().collect();

    Ok((
        Array1::from_vec(w_vec),
        rank,
        Array1::from_vec(singular_vec),
    ))
}

/// Solve the multi-output least squares problem `X @ W = Y` for `W`.
///
/// The 2-D companion to [`solve_lstsq`]: `Y` is `(n_samples, n_targets)` and
/// the returned solution `W` is `(n_features, n_targets)` — column `t` is the
/// minimum-norm least-squares solution of `X @ w = Y[:, t]`. Routes through the
/// same single-SVD [`ferray::linalg::lstsq`] (`ferray-linalg/src/solve.rs:208`),
/// which natively accepts a 2-D `b` (its `match b_shape.len()` arm) and returns
/// the `(n_features, n_targets)` solution row-major. This mirrors
/// scikit-learn's dense multi-output path `linalg.lstsq(X, Y)` with `Y` of
/// shape `(n_samples, n_targets)` (`sklearn/linear_model/_base.py:687`), where
/// the LAPACK-`gelsd` solve handles all targets in one SVD.
///
/// The same `Some(F::epsilon())` cutoff as the 1-D wrapper is used, pinning the
/// singular-value rank decision to scipy's `cond=eps` (`eps * s_max`).
///
/// Returns `(solution, rank, singular_values)`: the `(n_features, n_targets)`
/// minimum-norm solution, the effective rank of `X` (sklearn `rank_`), and the
/// singular values of `X` (sklearn `singular_`).
///
/// # Errors
///
/// Returns [`FerroError::NumericalInstability`] if the underlying SVD fails or
/// the ferray↔ndarray bridge encounters a shape inconsistency.
pub(crate) fn solve_lstsq_multi<F: LinalgFloat>(
    x: &Array2<F>,
    y: &Array2<F>,
) -> Result<(Array2<F>, usize, Array1<F>), FerroError> {
    let (n_samples, n_features) = x.dim();
    let n_targets = y.ncols();

    // Bridge ndarray -> ferray (R-SUBSTRATE-4): flat row-major Vec + shape.
    let x_flat: Vec<F> = x.iter().copied().collect();
    let a =
        FerrayArray::<F, ferray::Ix2>::from_vec(ferray::Ix2::new([n_samples, n_features]), x_flat)
            .map_err(|e| FerroError::NumericalInstability {
                message: format!("ferray lstsq_multi: failed to build design matrix: {e}"),
            })?;

    // Build the 2-D ferray `b` as `[n_samples, n_targets]` from Y's row-major
    // flat data; ferray's lstsq dispatches on `b_shape.len() == 2`.
    let y_flat: Vec<F> = y.iter().copied().collect();
    let b = FerrayArray::<F, IxDyn>::from_vec(IxDyn::new(&[n_samples, n_targets]), y_flat)
        .map_err(|e| FerroError::NumericalInstability {
            message: format!("ferray lstsq_multi: failed to build target matrix: {e}"),
        })?;

    // Single-SVD gelsd-equivalent solve for all targets at once. `Some(eps)`
    // pins the cutoff to scipy's `cond=eps` (see [`solve_lstsq`]).
    let (solution, _residuals, rank, singular) = ferray::linalg::lstsq(&a, &b, Some(F::epsilon()))
        .map_err(|e| FerroError::NumericalInstability {
            message: format!("ferray lstsq_multi solve failed: {e}"),
        })?;

    // Bridge the `(n_features, n_targets)` solution ferray -> ndarray. ferray
    // returns it as a 2-D `IxDyn` array, row-major; rebuild the owned
    // `Array2<F>` callers expect.
    let solution_nd = solution.into_ndarray();
    let sol_vec: Vec<F> = solution_nd.iter().copied().collect();
    if sol_vec.len() != n_features * n_targets {
        return Err(FerroError::NumericalInstability {
            message: format!(
                "ferray lstsq_multi: solution length {} does not match {} features x {} targets",
                sol_vec.len(),
                n_features,
                n_targets
            ),
        });
    }
    let coef = Array2::from_shape_vec((n_features, n_targets), sol_vec).map_err(|e| {
        FerroError::NumericalInstability {
            message: format!("ferray lstsq_multi: failed to reshape solution: {e}"),
        }
    })?;

    let singular_nd = singular.into_ndarray();
    let singular_vec: Vec<F> = singular_nd.iter().copied().collect();

    Ok((coef, rank, Array1::from_vec(singular_vec)))
}

/// Solve the non-negative least squares (NNLS) problem.
///
/// Returns `x ≥ 0` minimizing `||A·x − b||₂` via the classic Lawson-Hanson
/// active-set algorithm, mirroring `scipy.optimize.nnls`, which
/// scikit-learn calls for `LinearRegression(positive=True)`
/// (`self.coef_ = optimize.nnls(X, y)[0]`,
/// `sklearn/linear_model/_base.py:647`).
///
/// The passive-set unconstrained least-squares subproblems are solved by
/// [`solve_lstsq`] on the submatrix of `A`'s passive columns (the same
/// single-SVD gelsd-equivalent solver the unconstrained OLS path uses),
/// scattering the solution back into the length-`n` vector.
///
/// Outer iterations are capped at `3 * n` to guarantee termination (the
/// current best `x` is returned if the cap is reached); production never
/// panics or loops forever.
///
/// # Errors
///
/// Returns [`FerroError::NumericalInstability`] if a passive-set
/// least-squares solve fails.
pub(crate) fn nnls<F: LinalgFloat>(a: &Array2<F>, b: &Array1<F>) -> Result<Array1<F>, FerroError> {
    let (m, n) = a.dim();

    // x = 0 (length n); passive set P tracked as a boolean mask.
    let mut x = Array1::<F>::zeros(n);
    let mut passive = vec![false; n];

    // Tolerance ≈ 10·eps·||A||_inf·max(m, n), matching the scale of the
    // termination test in Lawson-Hanson / scipy's NNLS.
    let a_inf = a.iter().fold(<F as num_traits::Zero>::zero(), |acc, &v| {
        let av = <F as Float>::abs(v);
        if av > acc { av } else { acc }
    });
    let max_mn = <F as num_traits::NumCast>::from(m.max(n).max(1))
        .unwrap_or_else(<F as num_traits::One>::one);
    let ten = <F as num_traits::NumCast>::from(10).unwrap_or_else(<F as num_traits::One>::one);
    let mut tol = ten * <F as Float>::epsilon() * a_inf * max_mn;
    let tol_positive = tol > <F as num_traits::Zero>::zero() && <F as Float>::is_finite(tol);
    if !tol_positive {
        // Degenerate scale (all-zero A or non-finite): fall back to a fixed
        // small positive tolerance so the loop still terminates cleanly.
        tol = ten * <F as Float>::epsilon();
    }

    let at = a.t();
    let max_outer = 3 * n;

    for _ in 0..max_outer {
        // Gradient w = Aᵀ(b − A·x).
        let residual = b - &a.dot(&x);
        let w = at.dot(&residual);

        // Pick j* = argmax_{j ∉ P} w[j]; stop if none exceeds tol.
        let mut best_j: Option<usize> = None;
        let mut best_w = tol;
        for j in 0..n {
            if !passive[j] && w[j] > best_w {
                best_w = w[j];
                best_j = Some(j);
            }
        }
        let Some(jstar) = best_j else {
            break;
        };
        passive[jstar] = true;

        // Inner loop: solve unconstrained LS on the passive columns, moving
        // any column that goes non-positive back to the active set.
        loop {
            let passive_idx: Vec<usize> = (0..n).filter(|&j| passive[j]).collect();
            if passive_idx.is_empty() {
                break;
            }

            // Build A[:, P] and solve the unconstrained LS subproblem.
            let mut a_p = Array2::<F>::zeros((m, passive_idx.len()));
            for (col, &j) in passive_idx.iter().enumerate() {
                for row in 0..m {
                    a_p[[row, col]] = a[[row, j]];
                }
            }
            let (z_p, _rank, _singular) = solve_lstsq(&a_p, b)?;

            // Scatter z_P into a length-n z (0 for active columns).
            let mut z = Array1::<F>::zeros(n);
            for (col, &j) in passive_idx.iter().enumerate() {
                z[j] = z_p[col];
            }

            // If all passive components are strictly positive, accept z.
            let all_positive = passive_idx
                .iter()
                .all(|&j| z[j] > <F as num_traits::Zero>::zero());
            if all_positive {
                for &j in &passive_idx {
                    x[j] = z[j];
                }
                break;
            }

            // α = min_{j∈P, z[j] ≤ 0} x[j] / (x[j] − z[j]).
            let mut alpha = <F as Float>::infinity();
            for &j in &passive_idx {
                if z[j] <= <F as num_traits::Zero>::zero() {
                    let denom = x[j] - z[j];
                    if denom > <F as num_traits::Zero>::zero() {
                        let ratio = x[j] / denom;
                        if ratio < alpha {
                            alpha = ratio;
                        }
                    }
                }
            }
            if !<F as Float>::is_finite(alpha) {
                // No valid step (numerical edge): accept z's positive part
                // and stop to guarantee progress/termination.
                for &j in &passive_idx {
                    x[j] = if z[j] > <F as num_traits::Zero>::zero() {
                        z[j]
                    } else {
                        <F as num_traits::Zero>::zero()
                    };
                }
                break;
            }

            // x = x + α·(z − x).
            for j in 0..n {
                x[j] = x[j] + alpha * (z[j] - x[j]);
            }

            // Remove from P every column with x[j] ≈ 0.
            for &j in &passive_idx {
                if x[j] <= tol {
                    passive[j] = false;
                    x[j] = <F as num_traits::Zero>::zero();
                }
            }
        }
    }

    // Clamp any tiny residual negatives to exactly zero (non-negativity).
    let zero = <F as num_traits::Zero>::zero();
    x.mapv_inplace(|v| if v < zero { zero } else { v });
    Ok(x)
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
/// For `alpha > 0` the normal-equations matrix `X^T X + alpha * I` is
/// positive definite, so the Cholesky solve succeeds and the fallbacks
/// never fire. For `alpha = 0` on a rank-deficient `X`, `X^T X` is singular:
/// both the Cholesky and the Gaussian-elimination solves fail, and the
/// chain falls through to the minimum-norm least-squares solve on the
/// original `X`/`y` (LAPACK `gelsd` via [`solve_lstsq`]). This mirrors
/// scikit-learn's `'cholesky'` branch, which on a `linalg.LinAlgError`
/// (singular `X^T X`) switches to the SVD solver
/// (`sklearn/linear_model/_ridge.py:752-756`):
///
/// ```text
/// try:
///     coef = _solve_cholesky(X, y, alpha)
/// except linalg.LinAlgError:
///     # use SVD solver if matrix is singular
///     solver = "svd"
/// ```
///
/// scikit-learn's SVD solver returns the minimum-norm solution; for
/// `alpha = 0` (`X^T X + 0 * I = X^T X`) that coincides with the gelsd
/// minimum-norm least-squares solution of `X @ w = y`, which is exactly
/// what [`solve_lstsq`] computes. (For `alpha > 0` the PD Cholesky always
/// succeeds, so the lstsq branch is unreachable and behavior is unchanged.)
///
/// # Errors
///
/// Returns [`FerroError::NumericalInstability`] if every solve in the
/// chain fails (e.g. the underlying SVD itself fails).
pub(crate) fn solve_ridge<F: LinalgFloat>(
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
        xtx[[i, i]] += alpha;
    }

    cholesky_solve(&xtx, &xty)
        .or_else(|_| gaussian_solve(n, &xtx, &xty))
        // The lstsq fallback now returns (solution, rank, singular); the Ridge
        // path consumes only the coefficient solution.
        .or_else(|_| solve_lstsq(x, y).map(|(w, _rank, _singular)| w))
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
    use ndarray::array;

    #[test]
    fn nnls_matches_scipy() {
        // Live oracle (scipy.optimize.nnls, the solver sklearn uses for
        // LinearRegression(positive=True), `_base.py:647`):
        //   cd /tmp && python3 -c "import numpy as np; \
        //     from scipy.optimize import nnls; \
        //     X=np.array([[1.,1.],[1.,2.],[2.,1.],[3.,2.],[2.,3.]]); \
        //     y=np.array([1.,0.5,3.,5.,1.5]); \
        //     print([round(c,8) for c in nnls(X,y)[0]])"
        //   -> [1.34210526, 0.0]
        let x = array![[1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [3.0, 2.0], [2.0, 3.0]];
        let y = array![1.0, 0.5, 3.0, 5.0, 1.5];
        let res = nnls(&x, &y);
        assert!(res.is_ok());
        if let Ok(coef) = res {
            assert_eq!(coef.len(), 2);
            assert_relative_eq!(coef[0], 1.342_105_26, epsilon = 1e-6);
            assert_relative_eq!(coef[1], 0.0, epsilon = 1e-6);
            // Non-negativity contract.
            assert!(coef.iter().all(|&c| c >= 0.0));
        }
    }

    #[test]
    fn nnls_equals_ols_when_unconstrained_nonneg() {
        // When the unconstrained least-squares optimum is already
        // all-non-negative, NNLS must not distort it. Oracle:
        //   cd /tmp && python3 -c "import numpy as np; \
        //     from scipy.optimize import nnls; from scipy.linalg import lstsq; \
        //     X=np.array([[1.,0.],[0.,1.],[1.,1.]]); y=np.array([1.,2.,3.]); \
        //     print([round(c,8) for c in nnls(X,y)[0]], \
        //           [round(c,8) for c in lstsq(X,y)[0]])"
        //   -> [1.0, 2.0] [1.0, 2.0]
        let x = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let y = array![1.0, 2.0, 3.0];
        let res = nnls(&x, &y);
        assert!(res.is_ok());
        if let Ok(coef) = res {
            assert_relative_eq!(coef[0], 1.0, epsilon = 1e-8);
            assert_relative_eq!(coef[1], 2.0, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_solve_lstsq_simple() {
        // 2x = 4 -> x = 2
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = Array1::from_vec(vec![2.0, 4.0, 6.0]);
        let w = solve_lstsq(&x, &y).unwrap();
        assert_relative_eq!(w.0[0], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_solve_lstsq_multi() {
        // y = x1 + 2*x2
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let w = solve_lstsq(&x, &y).unwrap();
        assert_relative_eq!(w.0[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(w.0[1], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_solve_lstsq_multi_two_targets() {
        // Multi-output min-norm lstsq: solve X @ W = Y for a 2-target Y on a
        // full-rank X. Oracle (scipy.linalg.lstsq handles 2-D b):
        //   python3 -c "import numpy as np; from scipy.linalg import lstsq; \
        //     X=np.array([[1.,0.],[2.,1.],[3.,1.],[4.,2.],[5.,3.]]); \
        //     Y=np.array([[2.1,1.0],[3.9,2.1],[6.2,2.9],[7.7,4.2],[10.3,5.1]]); \
        //     print([[round(v,8) for v in r] for r in lstsq(X,Y)[0]])"
        //   -> [[2.0195122, 0.96097561], [-0.0097561, 0.1195122]]
        // (solution is (n_features, n_targets); column t solves X @ w = Y[:, t].)
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 0.0, 2.0, 1.0, 3.0, 1.0, 4.0, 2.0, 5.0, 3.0],
        )
        .unwrap();
        let y = Array2::from_shape_vec(
            (5, 2),
            vec![2.1, 1.0, 3.9, 2.1, 6.2, 2.9, 7.7, 4.2, 10.3, 5.1],
        )
        .unwrap();
        let (w, rank, sing) = solve_lstsq_multi(&x, &y).unwrap();
        assert_eq!(w.dim(), (2, 2));
        assert_eq!(rank, 2);
        assert_eq!(sing.len(), 2);
        assert_relative_eq!(w[[0, 0]], 2.019_512_2, epsilon = 1e-7);
        assert_relative_eq!(w[[1, 0]], -0.009_756_1, epsilon = 1e-7);
        assert_relative_eq!(w[[0, 1]], 0.960_975_61, epsilon = 1e-7);
        assert_relative_eq!(w[[1, 1]], 0.119_512_2, epsilon = 1e-7);
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
        assert_relative_eq!(w.0[0], 0.5, epsilon = 1e-10);
        assert_relative_eq!(w.0[1], 0.5, epsilon = 1e-10);
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
        assert_relative_eq!(w.0[0], -0.055_555_555_555_555_83, epsilon = 1e-8);
        assert_relative_eq!(w.0[1], 0.111_111_111_111_111_12, epsilon = 1e-8);
        assert_relative_eq!(w.0[2], 0.277_777_777_777_778, epsilon = 1e-8);
    }
}
