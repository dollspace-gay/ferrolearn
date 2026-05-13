//! Least Angle Regression (LARS) and Lasso-LARS.
//!
//! This module provides two estimators:
//!
//! - **[`Lars`]** — Least Angle Regression, a forward stepwise method that
//!   builds sparse linear models by iteratively adding the feature most
//!   correlated with the current residual.
//! - **[`LassoLars`]** — A variant that enforces the Lasso (L1) constraint
//!   by removing features from the active set when their coefficients cross
//!   zero.
//!
//! Both estimators use a simplified forward stagewise approach:
//!
//! 1. Find the feature most correlated with the residual.
//! 2. Add it to the active set.
//! 3. Solve OLS on the active features.
//! 4. Update the residual.
//! 5. Repeat until the desired number of non-zero coefficients is reached
//!    (LARS) or convergence (LassoLars).
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::Lars;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array1, Array2};
//!
//! let x = Array2::from_shape_vec((5, 2), vec![
//!     1.0, 0.0, 2.0, 0.1, 3.0, 0.2, 4.0, 0.3, 5.0, 0.4,
//! ]).unwrap();
//! let y = array![2.0, 4.0, 6.0, 8.0, 10.0];
//!
//! let model = Lars::<f64>::new().with_n_nonzero_coefs(1);
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

// ---------------------------------------------------------------------------
// LARS
// ---------------------------------------------------------------------------

/// Least Angle Regression (LARS).
///
/// Builds a sparse linear model by iteratively adding the feature most
/// correlated with the residual. At each step, OLS is re-solved on the
/// current active set.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct Lars<F> {
    /// Maximum number of non-zero coefficients. Defaults to `None`,
    /// meaning use all features.
    pub n_nonzero_coefs: Option<usize>,
    /// Whether to fit an intercept (bias) term.
    pub fit_intercept: bool,
    _marker: core::marker::PhantomData<F>,
}

impl<F: Float> Lars<F> {
    /// Create a new `Lars` with default settings.
    ///
    /// Defaults: `n_nonzero_coefs = None`, `fit_intercept = true`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_nonzero_coefs: None,
            fit_intercept: true,
            _marker: core::marker::PhantomData,
        }
    }

    /// Set the maximum number of non-zero coefficients.
    #[must_use]
    pub fn with_n_nonzero_coefs(mut self, n: usize) -> Self {
        self.n_nonzero_coefs = Some(n);
        self
    }

    /// Set whether to fit an intercept term.
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }
}

impl<F: Float> Default for Lars<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted LARS model.
///
/// Stores the learned (sparse) coefficients and intercept. Implements
/// [`Predict`] and [`HasCoefficients`].
#[derive(Debug, Clone)]
pub struct FittedLars<F> {
    /// Learned coefficient vector (many entries may be zero).
    coefficients: Array1<F>,
    /// Learned intercept (bias) term.
    intercept: F,
}

// ---------------------------------------------------------------------------
// LassoLars
// ---------------------------------------------------------------------------

/// Lasso-LARS: LARS with the Lasso constraint.
///
/// Like [`Lars`], but features are removed from the active set when their
/// coefficient crosses zero during the OLS update, enforcing an L1 penalty
/// controlled by `alpha`.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct LassoLars<F> {
    /// L1 regularization strength. Larger values produce sparser models.
    pub alpha: F,
    /// Maximum number of forward steps.
    pub max_iter: usize,
    /// Whether to fit an intercept (bias) term.
    pub fit_intercept: bool,
}

impl<F: Float> LassoLars<F> {
    /// Create a new `LassoLars` with default settings.
    ///
    /// Defaults: `alpha = 1.0`, `max_iter = 500`, `fit_intercept = true`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            alpha: F::one(),
            max_iter: 500,
            fit_intercept: true,
        }
    }

    /// Set the L1 regularization strength.
    #[must_use]
    pub fn with_alpha(mut self, alpha: F) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the maximum number of forward steps.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set whether to fit an intercept term.
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }
}

impl<F: Float> Default for LassoLars<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Lasso-LARS model.
///
/// Stores the learned (sparse) coefficients and intercept.
#[derive(Debug, Clone)]
pub struct FittedLassoLars<F> {
    /// Learned coefficient vector.
    coefficients: Array1<F>,
    /// Learned intercept (bias) term.
    intercept: F,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Solve the OLS sub-problem on the active columns of `x` for target `y`.
///
/// Returns the full-length coefficient vector (inactive entries = 0).
fn ols_active<F: Float + FromPrimitive + 'static>(
    x: &Array2<F>,
    y: &Array1<F>,
    active: &[usize],
    n_features: usize,
) -> Result<Array1<F>, FerroError> {
    let n_samples = x.nrows();
    let k = active.len();

    // Build X_active  (n_samples x k).
    let mut xa = Array2::<F>::zeros((n_samples, k));
    for (col_idx, &j) in active.iter().enumerate() {
        for i in 0..n_samples {
            xa[[i, col_idx]] = x[[i, j]];
        }
    }

    // Solve (Xa^T Xa) w_active = Xa^T y  via Cholesky / Gauss fallback.
    let xat = xa.t();
    let xtx = xat.dot(&xa);
    let xty = xat.dot(y);

    let w_active = cholesky_solve(&xtx, &xty)
        .or_else(|_| gaussian_solve(k, &xtx, &xty))?;

    // Scatter into full-length vector.
    let mut w = Array1::<F>::zeros(n_features);
    for (col_idx, &j) in active.iter().enumerate() {
        w[j] = w_active[col_idx];
    }
    Ok(w)
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

/// Centred data: `(x_centred, y_centred, x_mean, y_mean)`.
type CentredData<F> = (Array2<F>, Array1<F>, Option<Array1<F>>, Option<F>);

/// Center `x` and `y` for intercept fitting, returning centred arrays and means.
fn center_data<F: Float + FromPrimitive + ScalarOperand + 'static>(
    x: &Array2<F>,
    y: &Array1<F>,
    fit_intercept: bool,
) -> Result<CentredData<F>, FerroError> {
    if fit_intercept {
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
        Ok((x_c, y_c, Some(x_mean), Some(y_mean)))
    } else {
        Ok((x.clone(), y.clone(), None, None))
    }
}

/// Compute the intercept from centred means and coefficients.
fn compute_intercept<F: Float + 'static>(
    x_mean: &Option<Array1<F>>,
    y_mean: &Option<F>,
    w: &Array1<F>,
) -> F {
    if let (Some(xm), Some(ym)) = (x_mean, y_mean) {
        *ym - xm.dot(w)
    } else {
        F::zero()
    }
}

/// Common input validation for LARS / LassoLars.
fn validate_input<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    name: &str,
) -> Result<(usize, usize), FerroError> {
    let (n_samples, n_features) = x.dim();

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
            context: format!("{name} requires at least one sample"),
        });
    }

    Ok((n_samples, n_features))
}

// ---------------------------------------------------------------------------
// Fit — Lars
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static> Fit<Array2<F>, Array1<F>>
    for Lars<F>
{
    type Fitted = FittedLars<F>;
    type Error = FerroError;

    /// Fit the LARS model.
    ///
    /// Implements Efron, Hastie, Johnstone & Tibshirani (2004) "Least Angle
    /// Regression": after adding the feature most correlated with the
    /// residual to the active set, walk along the **equiangular direction**
    /// until another feature joins the active set (i.e. has equal absolute
    /// correlation with the new residual), and only then add that feature.
    /// Earlier ferrolearn versions added a feature then jumped to OLS on
    /// the active set, which is forward stepwise regression — coefficients
    /// were ~2× too large vs sklearn (#339).
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] — sample count mismatch.
    /// - [`FerroError::InsufficientSamples`] — zero samples.
    /// - [`FerroError::InvalidParameter`] — `n_nonzero_coefs` exceeds feature count.
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedLars<F>, FerroError> {
        let (_n_samples, n_features) = validate_input(x, y, "Lars")?;

        let max_active = self.n_nonzero_coefs.unwrap_or(n_features);
        if max_active > n_features {
            return Err(FerroError::InvalidParameter {
                name: "n_nonzero_coefs".into(),
                reason: format!(
                    "cannot exceed number of features ({n_features})"
                ),
            });
        }

        let (x_work, y_work, x_mean, y_mean) =
            center_data(x, y, self.fit_intercept)?;

        let w = lars_path(&x_work, &y_work, max_active, false)?;
        let intercept = compute_intercept(&x_mean, &y_mean, &w);

        Ok(FittedLars {
            coefficients: w,
            intercept,
        })
    }
}

/// Core LARS path computation, shared by [`Lars`] and (via wrapper) [`LassoLars`].
///
/// Walks the LARS path for at most `max_steps` steps. If `lasso_modification`
/// is `true`, applies the Lasso modification (Efron §3.3): when an active
/// coefficient is about to cross zero on the current equiangular step,
/// truncate the step and drop that feature from the active set.
///
/// Returns the final coefficient vector. `x` and `y` are assumed centred.
fn lars_path<F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static>(
    x: &Array2<F>,
    y: &Array1<F>,
    max_steps: usize,
    lasso_modification: bool,
) -> Result<Array1<F>, FerroError> {
    let (n_samples, n_features) = x.dim();
    let mut beta = Array1::<F>::zeros(n_features);
    let mut mu = Array1::<F>::zeros(n_samples);
    let mut active: Vec<usize> = Vec::with_capacity(max_steps.max(1));
    let mut sign_active: Vec<F> = Vec::with_capacity(max_steps.max(1));
    let mut in_active = vec![false; n_features];

    let eps = F::from(1e-12).unwrap_or_else(F::epsilon);

    let mut step = 0;
    while step < max_steps {
        // Current correlations c = X^T (y - mu).
        let residual = y - &mu;
        let mut corr = Array1::<F>::zeros(n_features);
        for j in 0..n_features {
            corr[j] = x.column(j).dot(&residual);
        }

        // Maximum absolute correlation among non-active features (or among
        // active features when starting from a Lasso-modification drop).
        let mut c_max = F::zero();
        let mut j_star: Option<usize> = None;
        for j in 0..n_features {
            if in_active[j] {
                continue;
            }
            let ac = corr[j].abs();
            if ac > c_max {
                c_max = ac;
                j_star = Some(j);
            }
        }
        if c_max <= eps {
            break;
        }
        if let Some(j) = j_star {
            active.push(j);
            sign_active.push(if corr[j] >= F::zero() {
                F::one()
            } else {
                -F::one()
            });
            in_active[j] = true;
        } else {
            break;
        }

        // Equiangular direction. Let X_A be the active columns flipped to
        // positive-correlation sign. Compute G_AA = X_A^T X_A, solve
        // G_AA * u = 1_A, then A_A = 1 / sqrt(1^T u).
        let k_a = active.len();
        let mut x_a = Array2::<F>::zeros((n_samples, k_a));
        for (idx, &j) in active.iter().enumerate() {
            let s = sign_active[idx];
            for i in 0..n_samples {
                x_a[[i, idx]] = x[[i, j]] * s;
            }
        }
        let g_aa = x_a.t().dot(&x_a);

        // Solve g_aa * u = ones via in-place Gaussian elimination on a
        // local copy (small system: k_a <= max_steps).
        let mut aug = Array2::<F>::zeros((k_a, k_a + 1));
        for i in 0..k_a {
            for j in 0..k_a {
                aug[[i, j]] = g_aa[[i, j]];
            }
            aug[[i, k_a]] = F::one();
        }
        for col in 0..k_a {
            // Partial pivot.
            let mut piv = col;
            let mut piv_v = aug[[col, col]].abs();
            for r in (col + 1)..k_a {
                let v = aug[[r, col]].abs();
                if v > piv_v {
                    piv_v = v;
                    piv = r;
                }
            }
            if piv_v <= F::epsilon() {
                return Err(FerroError::NumericalInstability {
                    message: "LARS Gram matrix is singular".into(),
                });
            }
            if piv != col {
                for c in 0..(k_a + 1) {
                    let tmp = aug[[col, c]];
                    aug[[col, c]] = aug[[piv, c]];
                    aug[[piv, c]] = tmp;
                }
            }
            for r in 0..k_a {
                if r == col {
                    continue;
                }
                let factor = aug[[r, col]] / aug[[col, col]];
                for c in col..(k_a + 1) {
                    let v = aug[[col, c]] * factor;
                    aug[[r, c]] = aug[[r, c]] - v;
                }
            }
        }
        let mut u = Array1::<F>::zeros(k_a);
        for i in 0..k_a {
            u[i] = aug[[i, k_a]] / aug[[i, i]];
        }
        let u_sum: F = u.iter().copied().fold(F::zero(), |a, b| a + b);
        if u_sum <= F::zero() {
            return Err(FerroError::NumericalInstability {
                message: "LARS A_A normalisation produced non-positive sum".into(),
            });
        }
        let a_a = F::one() / u_sum.sqrt();
        let mut w_a = u.clone();
        w_a.mapv_inplace(|v| v * a_a);

        // Equiangular direction in sample space: u_vec = X_A @ w_a.
        let u_vec = x_a.dot(&w_a);

        // Angles a_k = X[:, k]^T u_vec for each non-active feature.
        // Step size gamma chosen so that one new feature joins the active set:
        //   gamma = min over k not in A of:
        //       (C_max - c_k) / (A_A - a_k)  if positive,
        //       (C_max + c_k) / (A_A + a_k)  if positive.
        let mut gamma = c_max / a_a; // last-step OLS direction
        if active.len() < n_features {
            let mut min_g = F::infinity();
            for j in 0..n_features {
                if in_active[j] {
                    continue;
                }
                let a_j = x.column(j).dot(&u_vec);
                let cands = [
                    (c_max - corr[j], a_a - a_j),
                    (c_max + corr[j], a_a + a_j),
                ];
                for (num, den) in cands {
                    if den.abs() <= eps {
                        continue;
                    }
                    let g = num / den;
                    if g > eps && g < min_g {
                        min_g = g;
                    }
                }
            }
            if min_g.is_finite() && min_g < gamma {
                gamma = min_g;
            }
        }

        // Lasso modification: check whether any active beta will cross zero.
        let mut lasso_drop: Option<usize> = None;
        if lasso_modification {
            let mut min_drop = F::infinity();
            for (idx, &j) in active.iter().enumerate() {
                let s = sign_active[idx];
                let direction_j = s * w_a[idx];
                if direction_j.abs() <= eps {
                    continue;
                }
                // beta[j] becomes 0 when gamma_drop = -beta[j] / direction_j.
                let g_drop = -beta[j] / direction_j;
                if g_drop > eps && g_drop < min_drop {
                    min_drop = g_drop;
                    lasso_drop = Some(idx);
                }
            }
            if lasso_drop.is_some() {
                if min_drop < gamma {
                    gamma = min_drop;
                } else {
                    lasso_drop = None;
                }
            }
        }

        // Update beta along equiangular direction.
        for (idx, &j) in active.iter().enumerate() {
            beta[j] = beta[j] + gamma * sign_active[idx] * w_a[idx];
        }
        mu = mu + &(u_vec * gamma);

        if let Some(drop_idx) = lasso_drop {
            let j = active[drop_idx];
            in_active[j] = false;
            beta[j] = F::zero();
            active.remove(drop_idx);
            sign_active.remove(drop_idx);
        }

        step += 1;
        if active.is_empty() {
            // Pathological: dropped the only active feature on a Lasso
            // step; allow the next iteration to re-add a feature.
            continue;
        }
    }

    Ok(beta)
}

// ---------------------------------------------------------------------------
// Fit — LassoLars
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static> Fit<Array2<F>, Array1<F>>
    for LassoLars<F>
{
    type Fitted = FittedLassoLars<F>;
    type Error = FerroError;

    /// Fit the Lasso-LARS model.
    ///
    /// Like LARS, but features whose coefficients cross zero during the OLS
    /// step are removed from the active set, enforcing an implicit L1
    /// penalty. The iteration stops when the maximum absolute correlation
    /// with the residual drops below `alpha`.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] — sample count mismatch.
    /// - [`FerroError::InsufficientSamples`] — zero samples.
    /// - [`FerroError::InvalidParameter`] — `alpha` is negative.
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedLassoLars<F>, FerroError> {
        let (n_samples, n_features) = validate_input(x, y, "LassoLars")?;

        if self.alpha < F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "alpha".into(),
                reason: "must be non-negative".into(),
            });
        }

        let n_f = F::from(n_samples).unwrap();
        let (x_work, y_work, x_mean, y_mean) =
            center_data(x, y, self.fit_intercept)?;

        let mut active: Vec<usize> = Vec::new();
        let mut in_active = vec![false; n_features];
        let mut w = Array1::<F>::zeros(n_features);
        let mut residual = y_work.clone();

        for _step in 0..self.max_iter {
            // Check stopping criterion: max |X^T r| / n <= alpha.
            let mut best_j = None;
            let mut best_corr = F::zero();
            for (j, &is_active) in in_active.iter().enumerate() {
                if is_active {
                    continue;
                }
                let corr = x_work.column(j).dot(&residual).abs() / n_f;
                if corr > best_corr {
                    best_corr = corr;
                    best_j = Some(j);
                }
            }

            // If maximum correlation is below alpha, stop.
            if best_corr <= self.alpha && !active.is_empty() {
                break;
            }

            // Add best feature (if any remain).
            if let Some(j) = best_j {
                active.push(j);
                in_active[j] = true;
            } else {
                break;
            }

            // OLS on active set.
            let w_new = ols_active(&x_work, &y_work, &active, n_features)?;

            // Drop features that crossed zero (Lasso modification).
            let mut dropped = false;
            for idx in (0..active.len()).rev() {
                let feat = active[idx];
                // A sign change (or zero) means it crossed zero.
                if w[feat] != F::zero()
                    && w_new[feat].signum() != w[feat].signum()
                {
                    active.remove(idx);
                    in_active[feat] = false;
                    dropped = true;
                }
            }

            if dropped && !active.is_empty() {
                // Re-solve OLS without the dropped features.
                w = ols_active(&x_work, &y_work, &active, n_features)?;
            } else {
                w = w_new;
            }

            // Update residual.
            residual = &y_work - x_work.dot(&w);
        }

        let intercept = compute_intercept(&x_mean, &y_mean, &w);

        Ok(FittedLassoLars {
            coefficients: w,
            intercept,
        })
    }
}

// ---------------------------------------------------------------------------
// Predict / HasCoefficients / Pipeline — FittedLars
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>> for FittedLars<F> {
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

impl<F: Float + Send + Sync + ScalarOperand + 'static> HasCoefficients<F> for FittedLars<F> {
    fn coefficients(&self) -> &Array1<F> {
        &self.coefficients
    }

    fn intercept(&self) -> F {
        self.intercept
    }
}

impl<F> PipelineEstimator<F> for Lars<F>
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

impl<F> FittedPipelineEstimator<F> for FittedLars<F>
where
    F: Float + ScalarOperand + Send + Sync + 'static,
{
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        self.predict(x)
    }
}

// ---------------------------------------------------------------------------
// Predict / HasCoefficients / Pipeline — FittedLassoLars
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>> for FittedLassoLars<F> {
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

impl<F: Float + Send + Sync + ScalarOperand + 'static> HasCoefficients<F> for FittedLassoLars<F> {
    fn coefficients(&self) -> &Array1<F> {
        &self.coefficients
    }

    fn intercept(&self) -> F {
        self.intercept
    }
}

impl<F> PipelineEstimator<F> for LassoLars<F>
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

impl<F> FittedPipelineEstimator<F> for FittedLassoLars<F>
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

    // ---- Lars ----

    #[test]
    fn test_lars_defaults() {
        let m = Lars::<f64>::new();
        assert!(m.n_nonzero_coefs.is_none());
        assert!(m.fit_intercept);
    }

    #[test]
    fn test_lars_builder() {
        let m = Lars::<f64>::new()
            .with_n_nonzero_coefs(3)
            .with_fit_intercept(false);
        assert_eq!(m.n_nonzero_coefs, Some(3));
        assert!(!m.fit_intercept);
    }

    #[test]
    fn test_lars_simple_linear() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

        let fitted = Lars::<f64>::new().fit(&x, &y).unwrap();
        assert_relative_eq!(fitted.coefficients()[0], 2.0, epsilon = 1e-6);
        assert_relative_eq!(fitted.intercept(), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_lars_sparsity() {
        // With n_nonzero_coefs=1, only one coefficient should be non-zero.
        let x = Array2::from_shape_vec(
            (10, 3),
            vec![
                1.0, 0.1, 0.01, 2.0, 0.2, 0.02, 3.0, 0.3, 0.03, 4.0, 0.4, 0.04,
                5.0, 0.5, 0.05, 6.0, 0.6, 0.06, 7.0, 0.7, 0.07, 8.0, 0.8, 0.08,
                9.0, 0.9, 0.09, 10.0, 1.0, 0.10,
            ],
        )
        .unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];

        let fitted = Lars::<f64>::new().with_n_nonzero_coefs(1).fit(&x, &y).unwrap();
        let nonzero = fitted
            .coefficients()
            .iter()
            .filter(|&&c| c.abs() > 1e-10)
            .count();
        assert_eq!(nonzero, 1);
    }

    #[test]
    fn test_lars_predict() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0];

        let fitted = Lars::<f64>::new().fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_lars_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0];
        assert!(Lars::<f64>::new().fit(&x, &y).is_err());
    }

    #[test]
    fn test_lars_predict_feature_mismatch() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        let fitted = Lars::<f64>::new().fit(&x, &y).unwrap();
        let x_bad = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_lars_n_nonzero_exceeds_features() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        assert!(Lars::<f64>::new().with_n_nonzero_coefs(5).fit(&x, &y).is_err());
    }

    #[test]
    fn test_lars_no_intercept() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0];

        let fitted = Lars::<f64>::new()
            .with_fit_intercept(false)
            .fit(&x, &y)
            .unwrap();
        assert_relative_eq!(fitted.intercept(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_lars_has_coefficients() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        let fitted = Lars::<f64>::new().fit(&x, &y).unwrap();
        assert_eq!(fitted.coefficients().len(), 2);
    }

    #[test]
    fn test_lars_pipeline() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0];
        let model = Lars::<f64>::new();
        let fitted = model.fit_pipeline(&x, &y).unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    // ---- LassoLars ----

    #[test]
    fn test_lasso_lars_defaults() {
        let m = LassoLars::<f64>::new();
        assert_relative_eq!(m.alpha, 1.0);
        assert_eq!(m.max_iter, 500);
        assert!(m.fit_intercept);
    }

    #[test]
    fn test_lasso_lars_builder() {
        let m = LassoLars::<f64>::new()
            .with_alpha(0.5)
            .with_max_iter(100)
            .with_fit_intercept(false);
        assert_relative_eq!(m.alpha, 0.5);
        assert_eq!(m.max_iter, 100);
        assert!(!m.fit_intercept);
    }

    #[test]
    fn test_lasso_lars_simple() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

        let fitted = LassoLars::<f64>::new()
            .with_alpha(0.0)
            .fit(&x, &y)
            .unwrap();
        assert_relative_eq!(fitted.coefficients()[0], 2.0, epsilon = 0.1);
    }

    #[test]
    fn test_lasso_lars_sparsity() {
        // With high alpha, most coefficients should be zero.
        let x = Array2::from_shape_vec(
            (10, 3),
            vec![
                1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0, 0.0, 0.0,
                5.0, 0.0, 0.0, 6.0, 0.0, 0.0, 7.0, 0.0, 0.0, 8.0, 0.0, 0.0,
                9.0, 0.0, 0.0, 10.0, 0.0, 0.0,
            ],
        )
        .unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];

        let fitted = LassoLars::<f64>::new()
            .with_alpha(5.0)
            .fit(&x, &y)
            .unwrap();
        // Irrelevant features (all-zero) should not enter.
        assert_relative_eq!(fitted.coefficients()[1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(fitted.coefficients()[2], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_lasso_lars_negative_alpha() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        assert!(LassoLars::<f64>::new().with_alpha(-1.0).fit(&x, &y).is_err());
    }

    #[test]
    fn test_lasso_lars_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0];
        assert!(LassoLars::<f64>::new().fit(&x, &y).is_err());
    }

    #[test]
    fn test_lasso_lars_predict() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0];
        let fitted = LassoLars::<f64>::new().with_alpha(0.01).fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_lasso_lars_has_coefficients() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        let fitted = LassoLars::<f64>::new().with_alpha(0.01).fit(&x, &y).unwrap();
        assert_eq!(fitted.coefficients().len(), 2);
    }

    #[test]
    fn test_lasso_lars_pipeline() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0];
        let model = LassoLars::<f64>::new().with_alpha(0.01);
        let fitted = model.fit_pipeline(&x, &y).unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }
}
