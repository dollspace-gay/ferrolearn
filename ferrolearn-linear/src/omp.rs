//! Orthogonal Matching Pursuit (OMP).
//!
//! This module provides [`OrthogonalMatchingPursuit`], a greedy feature
//! selection algorithm that iteratively selects the feature most correlated
//! with the current residual, adds it to a support set, solves OLS on
//! the support, and updates the residual. The process repeats until the
//! desired number of non-zero coefficients is reached or the residual
//! tolerance is met.
//!
//! ## REQ status (per `.design/linear/omp.md`, mirrors `sklearn/linear_model/_omp.py` @ 1.5.2)
//!
//! Mirrors `sklearn.linear_model.OrthogonalMatchingPursuit` (`_omp.py:645`), greedy Cholesky OMP.
//! coef_/intercept_ match the live oracle to ~1e-12 on the diabetes dataset.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (greedy OMP fit) | SHIPPED | `Fit for OrthogonalMatchingPursuit`; `OMP(n_nonzero_coefs=5)` coef_/intercept_ match sklearn to 1e-12 on diabetes. Consumer: `pub use OrthogonalMatchingPursuit` (boundary API). |
//! | REQ-2 (default n_nonzero_coefs = max(int(0.1·n_features),1)) | SHIPPED | when both n_nonzero_coefs and tol are None, defaults to `max(int(0.1·n_features),1)` and fits (`_omp.py:785`). Closed #488 (was erroring). |
//! | REQ-3 (tol stopping ‖r‖²≤tol) | SHIPPED | residual-norm stopping (minor strict-before vs ≤-after boundary, equivalent for typical inputs). |
//! | REQ-4 (predict) | SHIPPED | `Predict for FittedOMP`. |
//! | REQ-5 (fit_intercept / HasCoefficients) | SHIPPED | centering + `HasCoefficients`. |
//! | REQ-6..10 NOT-STARTED | Gram/precompute path and `orthogonal_mp_gram` (#489), OrthogonalMatchingPursuitCV (#490), estimator `n_iter_` (#491), multi-output (#492), ferray substrate (#493). |
//! | REQ-11 (non-finite input rejected) | SHIPPED | `Fit::fit for OrthogonalMatchingPursuit` rejects any NaN/+/-inf in X or y BEFORE the greedy path with `FerroError::InvalidParameter`, mirroring sklearn's `_validate_data(force_all_finite=True)` (`_omp.py:772`) → `ValueError("Input X contains NaN.")` / `"... contains infinity ..."`. `.iter().any(|v| !v.is_finite())` catches both NaN and Inf; OMP takes no `sample_weight`; the finite path is byte-identical. Verified vs the live sklearn 1.5.2 oracle (R-CHAR-3): `OrthogonalMatchingPursuit().fit` raises `ValueError` for NaN/+inf/-inf in X and NaN/inf in y (`tests/divergence_linear_nonfinite_batch2.rs::omp_*`). Non-test consumer: the existing `Fit::fit` / `pub use OrthogonalMatchingPursuit` boundary consumers. (#2259) |
//! | REQ-12 (`orthogonal_mp` helper) | SHIPPED | `pub fn orthogonal_mp` exposes the dense single-output helper path with `n_iter` and optional coefficient path, reusing the estimator's greedy Cholesky OMP core. Oracle tests `orthogonal_mp_helper_*_matches_sklearn`. |
//!
//! acto-critic: the greedy path matches sklearn exactly (1e-12); the default-construction
//! divergence (#488 — errored where sklearn applies 0.1·n_features) found and fixed. Two states
//! only per goal.md R-DEFER-2.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::OrthogonalMatchingPursuit;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array1, Array2};
//!
//! let x = Array2::from_shape_vec((5, 3), vec![
//!     1.0, 0.0, 0.0,
//!     2.0, 0.1, 0.0,
//!     3.0, 0.0, 0.1,
//!     4.0, 0.1, 0.0,
//!     5.0, 0.0, 0.1,
//! ]).unwrap();
//! let y = array![2.0, 4.0, 6.0, 8.0, 10.0];
//!
//! let model = OrthogonalMatchingPursuit::<f64>::new().with_n_nonzero_coefs(1);
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 5);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasCoefficients;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, PipelineEstimator};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2, Axis, ScalarOperand};
use num_traits::{Float, FromPrimitive};

/// Orthogonal Matching Pursuit.
///
/// A greedy sparse approximation algorithm that selects features one at a
/// time. At each iteration it picks the feature most correlated with the
/// residual, adds it to the support, solves OLS on the support set, and
/// re-computes the residual.
///
/// Termination is controlled by either `n_nonzero_coefs` (maximum
/// support size) or `tol` (residual norm threshold), whichever is reached
/// first.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct OrthogonalMatchingPursuit<F> {
    /// Maximum number of non-zero coefficients. Defaults to `None` (use
    /// all features or stop at `tol`).
    pub n_nonzero_coefs: Option<usize>,
    /// Residual norm tolerance. If the squared residual norm drops below
    /// this threshold the algorithm terminates. Defaults to `None`.
    pub tol: Option<F>,
    /// Whether to fit an intercept (bias) term.
    pub fit_intercept: bool,
}

impl<F: Float> OrthogonalMatchingPursuit<F> {
    /// Create a new `OrthogonalMatchingPursuit` with default settings.
    ///
    /// Defaults: `n_nonzero_coefs = None`, `tol = None`,
    /// `fit_intercept = true`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_nonzero_coefs: None,
            tol: None,
            fit_intercept: true,
        }
    }

    /// Set the maximum number of non-zero coefficients.
    #[must_use]
    pub fn with_n_nonzero_coefs(mut self, n: usize) -> Self {
        self.n_nonzero_coefs = Some(n);
        self
    }

    /// Set the residual norm tolerance.
    #[must_use]
    pub fn with_tol(mut self, tol: F) -> Self {
        self.tol = Some(tol);
        self
    }

    /// Set whether to fit an intercept term.
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }
}

impl<F: Float> Default for OrthogonalMatchingPursuit<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Orthogonal Matching Pursuit model.
///
/// Stores the learned (sparse) coefficients and intercept.
#[derive(Debug, Clone)]
pub struct FittedOMP<F> {
    /// Learned coefficient vector (many entries may be zero).
    coefficients: Array1<F>,
    /// Learned intercept (bias) term.
    intercept: F,
}

/// Options for [`orthogonal_mp`].
#[derive(Debug, Clone, Copy)]
pub struct OrthogonalMpOptions<F> {
    /// Desired number of non-zero coefficients.
    pub n_nonzero_coefs: Option<usize>,
    /// Maximum squared residual norm.
    pub tol: Option<F>,
    /// Whether to capture the coefficient path after each active-set update.
    pub return_path: bool,
}

impl<F: Float> OrthogonalMpOptions<F> {
    /// Create default options matching sklearn's dense helper defaults:
    /// `n_nonzero_coefs=None`, `tol=None`, and `return_path=False`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_nonzero_coefs: None,
            tol: None,
            return_path: false,
        }
    }

    /// Set the desired number of non-zero coefficients.
    #[must_use]
    pub fn with_n_nonzero_coefs(mut self, n_nonzero_coefs: Option<usize>) -> Self {
        self.n_nonzero_coefs = n_nonzero_coefs;
        self
    }

    /// Set the maximum squared residual norm.
    #[must_use]
    pub fn with_tol(mut self, tol: Option<F>) -> Self {
        self.tol = tol;
        self
    }

    /// Set whether to capture the coefficient path.
    #[must_use]
    pub fn with_return_path(mut self, return_path: bool) -> Self {
        self.return_path = return_path;
        self
    }
}

impl<F: Float> Default for OrthogonalMpOptions<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Result returned by [`orthogonal_mp`].
#[derive(Debug, Clone)]
pub struct OrthogonalMpResult<F> {
    coefficients: Array1<F>,
    path: Option<Array2<F>>,
    n_iter: usize,
}

impl<F: Float> OrthogonalMpResult<F> {
    /// Borrow the final OMP coefficient vector.
    #[must_use]
    pub fn coefficients(&self) -> &Array1<F> {
        &self.coefficients
    }

    /// Borrow the optional coefficient path, shaped `(n_features, n_iter)`.
    #[must_use]
    pub fn path(&self) -> Option<&Array2<F>> {
        self.path.as_ref()
    }

    /// Return the number of active-set iterations.
    #[must_use]
    pub fn n_iter(&self) -> usize {
        self.n_iter
    }
}

/// Solve a dense single-output Orthogonal Matching Pursuit problem.
///
/// This is the Rust analogue of `sklearn.linear_model.orthogonal_mp` for dense
/// `ndarray` inputs and a one-dimensional target. Inputs are used as provided:
/// no intercept is fitted and no centering is performed. The returned result
/// always includes the final coefficients and active-set iteration count; when
/// `return_path=true`, it also includes the full coefficient path with one
/// column per active-set update.
///
/// # Errors
///
/// Returns [`FerroError`] for inconsistent shapes, empty input, non-finite
/// values, invalid `n_nonzero_coefs`/`tol`, or active-set solve failures.
pub fn orthogonal_mp<F>(
    x: &Array2<F>,
    y: &Array1<F>,
    options: OrthogonalMpOptions<F>,
) -> Result<OrthogonalMpResult<F>, FerroError>
where
    F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static,
{
    validate_omp_inputs(x, y)?;
    if let Some(tol) = options.tol
        && tol < F::zero()
    {
        return Err(FerroError::InvalidParameter {
            name: "tol".into(),
            reason: "must be non-negative".into(),
        });
    }
    let solved = solve_omp(
        x,
        y,
        options.n_nonzero_coefs,
        options.tol,
        options.return_path,
        false,
    )?;

    Ok(OrthogonalMpResult {
        coefficients: solved.coefficients,
        path: solved.path,
        n_iter: solved.n_iter,
    })
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

struct OmpSolved<F> {
    coefficients: Array1<F>,
    path: Option<Array2<F>>,
    n_iter: usize,
}

fn validate_omp_inputs<F: Float>(x: &Array2<F>, y: &Array1<F>) -> Result<(), FerroError> {
    let n_samples = x.nrows();
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
            context: "OMP requires at least one sample".into(),
        });
    }

    if x.iter().any(|v| !v.is_finite()) {
        return Err(FerroError::InvalidParameter {
            name: "X".into(),
            reason: "Input X contains NaN or infinity.".into(),
        });
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(FerroError::InvalidParameter {
            name: "y".into(),
            reason: "Input y contains NaN or infinity.".into(),
        });
    }

    Ok(())
}

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

/// Solve OLS on the active columns, returning the full-length coefficient vector.
fn ols_active<F: Float + FromPrimitive + 'static>(
    x: &Array2<F>,
    y: &Array1<F>,
    support: &[usize],
    n_features: usize,
) -> Result<Array1<F>, FerroError> {
    let n_samples = x.nrows();
    let k = support.len();

    let mut xa = Array2::<F>::zeros((n_samples, k));
    for (col_idx, &j) in support.iter().enumerate() {
        for i in 0..n_samples {
            xa[[i, col_idx]] = x[[i, j]];
        }
    }

    let xat = xa.t();
    let xtx = xat.dot(&xa);
    let xty = xat.dot(y);

    let w_active = cholesky_solve(&xtx, &xty).or_else(|_| gaussian_solve(k, &xtx, &xty))?;

    let mut w = Array1::<F>::zeros(n_features);
    for (col_idx, &j) in support.iter().enumerate() {
        w[j] = w_active[col_idx];
    }
    Ok(w)
}

fn solve_omp<F>(
    x_work: &Array2<F>,
    y_work: &Array1<F>,
    n_nonzero_coefs: Option<usize>,
    tol: Option<F>,
    return_path: bool,
    strict_n_nonzero_limit: bool,
) -> Result<OmpSolved<F>, FerroError>
where
    F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static,
{
    let n_features = x_work.ncols();

    // Default for n_nonzero_coefs when neither stopping criterion is set:
    // sklearn `_omp.py:123` / estimator `_omp.py:785` use
    // `max(int(0.1 * n_features), 1)`.
    let effective_n_nonzero = if n_nonzero_coefs.is_none() && tol.is_none() {
        Some(((n_features as f64 * 0.1) as usize).max(1))
    } else {
        n_nonzero_coefs
    };

    // The estimator historically rejects any explicit over-large support
    // request; sklearn's free helper only rejects it when no tol is supplied.
    if let Some(n) = n_nonzero_coefs
        && n > n_features
        && (strict_n_nonzero_limit || tol.is_none())
    {
        return Err(FerroError::InvalidParameter {
            name: "n_nonzero_coefs".into(),
            reason: format!("cannot exceed number of features ({n_features})"),
        });
    }

    let max_k = effective_n_nonzero.unwrap_or(n_features).min(n_features);
    let mut support: Vec<usize> = Vec::with_capacity(max_k);
    let mut in_support = vec![false; n_features];
    let mut w = Array1::<F>::zeros(n_features);
    let mut residual = y_work.clone();
    let mut path_steps = if return_path { Some(Vec::new()) } else { None };

    for _step in 0..max_k {
        // Find feature most correlated with residual.
        let mut best_j = None;
        let mut best_corr = F::zero();
        for (j, &is_in_support) in in_support.iter().enumerate() {
            if is_in_support {
                continue;
            }
            let corr = x_work.column(j).dot(&residual).abs();
            if corr > best_corr {
                best_corr = corr;
                best_j = Some(j);
            }
        }

        let j = match best_j {
            Some(j) => j,
            None => break,
        };

        support.push(j);
        in_support[j] = true;

        // OLS on support set.
        w = ols_active(x_work, y_work, &support, n_features)?;

        if let Some(steps) = &mut path_steps {
            steps.push(w.clone());
        }

        // Update residual.
        residual = y_work - &x_work.dot(&w);

        // sklearn checks the residual norm after each active-set update
        // (`_omp.py:141`) and stops on `<= tol`.
        if let Some(tol_val) = tol {
            let res_norm_sq = residual.dot(&residual);
            if res_norm_sq <= tol_val {
                break;
            }
        }
    }

    let path = path_steps.map(|steps| {
        let n_steps = steps.len();
        let mut path = Array2::<F>::zeros((n_features, n_steps));
        for (step, coef) in steps.iter().enumerate() {
            path.column_mut(step).assign(coef);
        }
        path
    });

    Ok(OmpSolved {
        coefficients: w,
        path,
        n_iter: support.len(),
    })
}

// ---------------------------------------------------------------------------
// Fit
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static> Fit<Array2<F>, Array1<F>>
    for OrthogonalMatchingPursuit<F>
{
    type Fitted = FittedOMP<F>;
    type Error = FerroError;

    /// Fit the OMP model.
    ///
    /// Greedily selects features by correlation with the residual and
    /// solves OLS on the growing support set.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] — sample count mismatch.
    /// - [`FerroError::InsufficientSamples`] — zero samples.
    /// - [`FerroError::InvalidParameter`] — `n_nonzero_coefs` exceeds features.
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedOMP<F>, FerroError> {
        // Non-finite input validation (#2259). sklearn
        // `OrthogonalMatchingPursuit.fit` ->
        // `self._validate_data(X, y, multi_output=True, y_numeric=True)`
        // (`_omp.py:772`) keeps the default `force_all_finite=True`, so
        // `check_array` rejects any NaN or +/-inf in X OR y with a `ValueError`
        // BEFORE the greedy path runs. `.iter().any(|v| !v.is_finite())` rejects
        // both NaN and Inf (bounds-safe, no panic, R-CODE-2). `OrthogonalMatching
        // Pursuit.fit` takes no `sample_weight`. The finite path is byte-identical
        // (the guard never fires on finite input).
        validate_omp_inputs(x, y)?;

        // Center data if fitting intercept.
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

        let solved = solve_omp(
            &x_work,
            &y_work,
            self.n_nonzero_coefs,
            self.tol,
            false,
            true,
        )?;
        let w = solved.coefficients;

        let intercept = if let (Some(xm), Some(ym)) = (&x_mean, &y_mean) {
            *ym - xm.dot(&w)
        } else {
            F::zero()
        };

        Ok(FittedOMP {
            coefficients: w,
            intercept,
        })
    }
}

// ---------------------------------------------------------------------------
// Predict / HasCoefficients / Pipeline
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>> for FittedOMP<F> {
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
        if x.ncols() != self.coefficients.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.coefficients.len()],
                actual: vec![x.ncols()],
                context: "number of features must match fitted model".into(),
            });
        }
        Ok(x.dot(&self.coefficients) + self.intercept)
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> HasCoefficients<F> for FittedOMP<F> {
    fn coefficients(&self) -> &Array1<F> {
        &self.coefficients
    }

    fn intercept(&self) -> F {
        self.intercept
    }
}

impl<F> PipelineEstimator<F> for OrthogonalMatchingPursuit<F>
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

impl<F> FittedPipelineEstimator<F> for FittedOMP<F>
where
    F: Float + ScalarOperand + Send + Sync + 'static,
{
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        self.predict(x)
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

    #[test]
    fn test_defaults() {
        let m = OrthogonalMatchingPursuit::<f64>::new();
        assert!(m.n_nonzero_coefs.is_none());
        assert!(m.tol.is_none());
        assert!(m.fit_intercept);
    }

    #[test]
    fn test_builder() {
        let m = OrthogonalMatchingPursuit::<f64>::new()
            .with_n_nonzero_coefs(3)
            .with_tol(1e-4)
            .with_fit_intercept(false);
        assert_eq!(m.n_nonzero_coefs, Some(3));
        assert_relative_eq!(m.tol.unwrap(), 1e-4);
        assert!(!m.fit_intercept);
    }

    #[test]
    fn test_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0];
        assert!(
            OrthogonalMatchingPursuit::<f64>::new()
                .with_n_nonzero_coefs(1)
                .fit(&x, &y)
                .is_err()
        );
    }

    #[test]
    fn test_default_n_nonzero_fits() {
        // sklearn `_omp.py:785`: when both n_nonzero_coefs and tol are None,
        // n_nonzero_coefs_ = max(int(0.1 * n_features), 1), and fit succeeds.
        // With 1 feature: max(int(0.1), 1) = 1.
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]);
        let y = array![1.0, 2.0, 3.0];
        assert!(x.is_ok(), "valid shape");
        let Ok(x) = x else { return };
        let result = OrthogonalMatchingPursuit::<f64>::new().fit(&x, &y);
        assert!(result.is_ok(), "default OMP must fit, not error");
        let Ok(fitted) = result else { return };
        let nonzero = fitted
            .coefficients()
            .iter()
            .filter(|&&c| c.abs() > 1e-10)
            .count();
        assert_eq!(nonzero, 1);
    }

    #[test]
    fn test_n_nonzero_exceeds_features() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        assert!(
            OrthogonalMatchingPursuit::<f64>::new()
                .with_n_nonzero_coefs(5)
                .fit(&x, &y)
                .is_err()
        );
    }

    #[test]
    fn test_simple_linear() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

        let fitted = OrthogonalMatchingPursuit::<f64>::new()
            .with_n_nonzero_coefs(1)
            .fit(&x, &y)
            .unwrap();
        assert_relative_eq!(fitted.coefficients()[0], 2.0, epsilon = 1e-6);
        assert_relative_eq!(fitted.intercept(), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_sparsity() {
        // With n_nonzero_coefs=1, only one coefficient should be non-zero.
        let x = Array2::from_shape_vec(
            (10, 3),
            vec![
                1.0, 0.1, 0.01, 2.0, 0.2, 0.02, 3.0, 0.3, 0.03, 4.0, 0.4, 0.04, 5.0, 0.5, 0.05,
                6.0, 0.6, 0.06, 7.0, 0.7, 0.07, 8.0, 0.8, 0.08, 9.0, 0.9, 0.09, 10.0, 1.0, 0.10,
            ],
        )
        .unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];

        let fitted = OrthogonalMatchingPursuit::<f64>::new()
            .with_n_nonzero_coefs(1)
            .fit(&x, &y)
            .unwrap();
        let nonzero = fitted
            .coefficients()
            .iter()
            .filter(|&&c| c.abs() > 1e-10)
            .count();
        assert_eq!(nonzero, 1);
    }

    #[test]
    fn test_tol_stopping() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0]; // perfect linear

        let fitted = OrthogonalMatchingPursuit::<f64>::new()
            .with_tol(1e-10)
            .fit(&x, &y)
            .unwrap();
        // Should find perfect fit with 1 feature.
        let preds = fitted.predict(&x).unwrap();
        for (pred, actual) in preds.iter().zip(y.iter()) {
            assert_relative_eq!(pred, actual, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_predict() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0];

        let fitted = OrthogonalMatchingPursuit::<f64>::new()
            .with_n_nonzero_coefs(1)
            .fit(&x, &y)
            .unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_predict_feature_mismatch() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        let fitted = OrthogonalMatchingPursuit::<f64>::new()
            .with_n_nonzero_coefs(1)
            .fit(&x, &y)
            .unwrap();
        let x_bad = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_has_coefficients() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        let fitted = OrthogonalMatchingPursuit::<f64>::new()
            .with_n_nonzero_coefs(2)
            .fit(&x, &y)
            .unwrap();
        assert_eq!(fitted.coefficients().len(), 2);
    }

    #[test]
    fn test_no_intercept() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0];

        let fitted = OrthogonalMatchingPursuit::<f64>::new()
            .with_n_nonzero_coefs(1)
            .with_fit_intercept(false)
            .fit(&x, &y)
            .unwrap();
        assert_relative_eq!(fitted.intercept(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pipeline() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0];
        let model = OrthogonalMatchingPursuit::<f64>::new().with_n_nonzero_coefs(1);
        let fitted = model.fit_pipeline(&x, &y).unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_multivariate_recovery() {
        // y = 1*x1 + 3*x2, OMP with n_nonzero_coefs=2 should recover both.
        let x = Array2::from_shape_vec(
            (5, 3),
            vec![
                1.0, 0.0, 0.5, 0.0, 1.0, 0.3, 1.0, 1.0, 0.1, 2.0, 0.0, 0.8, 0.0, 2.0, 0.4,
            ],
        )
        .unwrap();
        let y = array![1.0, 3.0, 4.0, 2.0, 6.0]; // = x1 + 3*x2

        let fitted = OrthogonalMatchingPursuit::<f64>::new()
            .with_n_nonzero_coefs(2)
            .fit(&x, &y)
            .unwrap();

        // The third feature should remain approximately zero.
        assert!(
            fitted.coefficients()[2].abs() < 0.5,
            "irrelevant feature should have near-zero coefficient"
        );
    }
}
