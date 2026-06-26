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
//! Both estimators follow the LARS equiangular homotopy: at each step the
//! direction is equiangular to all active features; `Lars` adds variables until
//! `n_nonzero_coefs` is reached, while `LassoLars` additionally applies the
//! Efron §3.3 drop condition (a variable leaves the active set when its
//! coefficient would cross zero) and stops at the target `alpha`.
//!
//! ## REQ status (per `.design/linear/lars.md`, mirrors `sklearn/linear_model/_least_angle.py` @ 1.5.2)
//!
//! Mirrors `sklearn.linear_model.Lars` / `LassoLars` (`_least_angle.py`). The LARS-Lasso path
//! follows sklearn's `_lars_path_solver` (equiangular direction + drop condition + alpha
//! interpolation); coef_/intercept_/active-set match the live oracle on the diabetes dataset.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (Lars method='lar' path) | SHIPPED | `compute_lars_path` (equiangular); `Lars(n_nonzero_coefs=5)` coef_/intercept_ match sklearn EXACTLY (1e-6) on diabetes. Consumer: `pub use Lars` (boundary API). |
//! | REQ-2 (LassoLars method='lasso' path) | SHIPPED | `LassoLars::fit` → `compute_lars_path` lasso branch (drop condition `z=-coef/least_squares`, alpha_min stopping, interpolation; `_least_angle.py:413+`). coef_/active-set match oracle at alpha=0.1/0.5/1.0. Closed #482 (was forward-stepwise OLS). |
//! | REQ-3 (predict) | SHIPPED | `Predict for FittedLars`/`FittedLassoLars`. |
//! | REQ-4 (fit_intercept / HasCoefficients) | SHIPPED | centering + `HasCoefficients`. |
//! | REQ-5..8 NOT-STARTED | fitted coef_path_/alphas_/active_/n_iter_ attrs (#483), constructor param parity (#484), LarsCV/LassoLarsCV (#485, separate units), ferray substrate (#486; path on ndarray/faer). |
//! | REQ-9 (Lars/LassoLars non-finite input rejected) | SHIPPED | `Fit::fit for Lars` AND the SEPARATE `Fit::fit for LassoLars` impl both reject any NaN/+/-inf in X or y BEFORE the LARS path with `FerroError::InvalidParameter`, mirroring sklearn's `_validate_data(force_all_finite=True)` (`Lars._fit` `_least_angle.py:1183`; `LassoLars.fit` override `_least_angle.py:1698` → `:1726`) → `ValueError("Input X contains NaN.")` / `"... contains infinity ..."`. `.iter().any(|v| !v.is_finite())` catches both NaN and Inf; neither `Lars.fit` nor `LassoLars.fit` takes a `sample_weight`; the finite path is byte-identical. Verified vs the live sklearn 1.5.2 oracle (R-CHAR-3): `Lars().fit` / `LassoLars(alpha=0.1).fit` raise `ValueError` for NaN/+inf/-inf in X and NaN/inf in y (`tests/divergence_linear_nonfinite_batch2.rs::lars_*`, `tests/divergence_lasso_lars_nonfinite.rs::lasso_lars_rejects_non_finite_input_like_sklearn`). Non-test consumer: the existing `Fit::fit` / `pub use Lars, LassoLars` boundary consumers. (#2259, #2260) |
//! | REQ-10 (`lars_path` public helper) | SHIPPED | `pub fn lars_path` + `LarsPathOptions` / `LarsPathResult` expose the dense single-output Rust analogue of `sklearn.linear_model.lars_path`, returning alphas, final active indices, coefficient path `(n_features, n_alphas)`, and `n_iter`. Oracle tests `lars_path_*_matches_sklearn`. |
//! | REQ-11 (`lars_path_gram` public helper) | SHIPPED | `pub fn lars_path_gram` exposes the sufficient-statistics analogue of `sklearn.linear_model.lars_path_gram`, taking `Xy`, `Gram`, and `n_samples`, and returning the same `LarsPathResult` surface. Oracle tests `lars_path_gram_*_matches_sklearn`. |
//! | REQ-12 (`LassoLarsIC`) | SHIPPED | `LassoLarsIC` / `FittedLassoLarsIC` compute the LARS-Lasso path and select the minimum AIC/BIC criterion knot (`_least_angle.py:2268-2308`), including OLS-based default noise-variance estimation (`:2312-2337`). Verification: `tests/divergence_lasso_lars_ic.rs` pins explicit-variance AIC/BIC and estimated-noise fits to the live sklearn 1.5.2 oracle; `tests/api_proof.rs` covers the crate-root API. |
//!
//! acto-critic + builder: `Lars` matched sklearn exactly; `LassoLars` diverged (used forward-stepwise
//! OLS instead of the equiangular lasso path) — rewritten to sklearn's `_lars_path_solver` lasso
//! branch (3c2c746). coef_/active-set now match the live oracle. Two states only per R-DEFER-2.
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

use ferray::linalg::LinalgFloat;
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

/// Information criterion used by [`LassoLarsIC`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LassoLarsICCriterion {
    /// Akaike information criterion.
    Aic,
    /// Bayesian information criterion.
    Bic,
}

/// Lasso-LARS model selected by AIC or BIC.
///
/// This mirrors sklearn's dense single-output `LassoLarsIC`: it computes the
/// full LARS-Lasso path, evaluates the information criterion at each knot, and
/// selects the alpha with the smallest criterion value. The Rust surface omits
/// sklearn's sparse/object inputs, `positive`, `precompute`, `copy_X`, `eps`,
/// Python warnings, and metadata routing.
#[derive(Debug, Clone)]
pub struct LassoLarsIC<F> {
    /// Information criterion used to choose the path knot.
    pub criterion: LassoLarsICCriterion,
    /// Maximum number of LARS path iterations.
    pub max_iter: usize,
    /// Whether to fit an intercept (bias) term.
    pub fit_intercept: bool,
    /// Optional externally supplied noise variance. When `None`, an OLS
    /// variance estimate is computed, matching sklearn's default.
    pub noise_variance: Option<F>,
}

impl<F: Float> LassoLarsIC<F> {
    /// Create a new `LassoLarsIC` with sklearn-compatible defaults:
    /// `criterion=AIC`, `max_iter=500`, `fit_intercept=true`,
    /// `noise_variance=None`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            criterion: LassoLarsICCriterion::Aic,
            max_iter: 500,
            fit_intercept: true,
            noise_variance: None,
        }
    }

    /// Set the information criterion.
    #[must_use]
    pub fn with_criterion(mut self, criterion: LassoLarsICCriterion) -> Self {
        self.criterion = criterion;
        self
    }

    /// Set the maximum number of LARS path iterations.
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

    /// Set an externally supplied noise variance.
    #[must_use]
    pub fn with_noise_variance(mut self, noise_variance: F) -> Self {
        self.noise_variance = Some(noise_variance);
        self
    }
}

impl<F: Float> Default for LassoLarsIC<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted [`LassoLarsIC`] model.
#[derive(Debug, Clone)]
pub struct FittedLassoLarsIC<F> {
    coefficients: Array1<F>,
    intercept: F,
    alpha: F,
    alphas: Array1<F>,
    criterion: Array1<F>,
    noise_variance: F,
    n_iter: usize,
}

impl<F: Float> FittedLassoLarsIC<F> {
    /// Selected alpha value.
    #[must_use]
    pub fn alpha(&self) -> F {
        self.alpha
    }

    /// Path alpha values.
    #[must_use]
    pub fn alphas(&self) -> &Array1<F> {
        &self.alphas
    }

    /// Criterion values aligned to [`alphas`](Self::alphas).
    #[must_use]
    pub fn criterion(&self) -> &Array1<F> {
        &self.criterion
    }

    /// Noise variance used to compute the criterion.
    #[must_use]
    pub fn noise_variance(&self) -> F {
        self.noise_variance
    }

    /// Number of LARS path iterations.
    #[must_use]
    pub fn n_iter(&self) -> usize {
        self.n_iter
    }
}

/// Path variant solved by [`lars_path`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LarsPathMethod {
    /// Least Angle Regression path (`method="lar"` in sklearn).
    Lar,
    /// Lasso path solved by LARS with coefficient-drop steps
    /// (`method="lasso"` in sklearn).
    Lasso,
}

/// Options for [`lars_path`].
#[derive(Debug, Clone)]
pub struct LarsPathOptions<F> {
    /// Maximum number of LARS path iterations.
    pub max_iter: usize,
    /// Minimum alpha/correlation along the path.
    pub alpha_min: F,
    /// Path variant to solve.
    pub method: LarsPathMethod,
}

impl<F: Float> LarsPathOptions<F> {
    /// Create default options matching sklearn's dense helper defaults for the
    /// supported Rust surface: `max_iter=500`, `alpha_min=0`, `method=Lar`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_iter: 500,
            alpha_min: F::zero(),
            method: LarsPathMethod::Lar,
        }
    }

    /// Set the maximum number of LARS iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the minimum alpha/correlation along the path.
    #[must_use]
    pub fn with_alpha_min(mut self, alpha_min: F) -> Self {
        self.alpha_min = alpha_min;
        self
    }

    /// Select the LARS path variant.
    #[must_use]
    pub fn with_method(mut self, method: LarsPathMethod) -> Self {
        self.method = method;
        self
    }
}

impl<F: Float> Default for LarsPathOptions<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Result returned by [`lars_path`].
#[derive(Debug, Clone)]
pub struct LarsPathResult<F> {
    alphas: Array1<F>,
    active: Vec<usize>,
    coefficients: Array2<F>,
    n_iter: usize,
}

impl<F: Float> LarsPathResult<F> {
    /// Borrow the alpha values at each path knot.
    #[must_use]
    pub fn alphas(&self) -> &Array1<F> {
        &self.alphas
    }

    /// Borrow the active feature indices at the end of the path.
    #[must_use]
    pub fn active(&self) -> &[usize] {
        &self.active
    }

    /// Borrow the coefficient path, shaped `(n_features, n_alphas)`.
    #[must_use]
    pub fn coefficients(&self) -> &Array2<F> {
        &self.coefficients
    }

    /// Number of LARS iterations run.
    #[must_use]
    pub fn n_iter(&self) -> usize {
        self.n_iter
    }
}

/// Compute a dense single-output LARS or LARS-Lasso path.
///
/// This is the Rust analogue of `sklearn.linear_model.lars_path` for dense
/// `ndarray` inputs and a one-dimensional target. Inputs are used as provided:
/// no intercept is fitted and no centering is performed. Callers that need an
/// intercept should center `X` and `y` before calling this helper, matching
/// sklearn's path-helper contract.
///
/// # Errors
///
/// Returns [`FerroError`] for inconsistent shapes, empty input, non-finite
/// values, invalid `alpha_min`, singular active-set Gram matrices, or other
/// numerical failures in the path solver.
pub fn lars_path<F>(
    x: &Array2<F>,
    y: &Array1<F>,
    options: LarsPathOptions<F>,
) -> Result<LarsPathResult<F>, FerroError>
where
    F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static,
{
    validate_input(x, y, "lars_path")?;
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
    if options.alpha_min < F::zero() || !options.alpha_min.is_finite() {
        return Err(FerroError::InvalidParameter {
            name: "alpha_min".into(),
            reason: "must be finite and non-negative".into(),
        });
    }

    let path = compute_lars_path(
        x,
        y,
        options.max_iter,
        options.method == LarsPathMethod::Lasso,
        options.alpha_min,
        None,
    )?;

    let n_features = x.ncols();
    let n_alphas = path.alphas.len();
    let mut coefficients = Array2::<F>::zeros((n_features, n_alphas));
    for (alpha_idx, coef) in path.coefficients.iter().enumerate() {
        coefficients.column_mut(alpha_idx).assign(coef);
    }

    Ok(LarsPathResult {
        alphas: Array1::from_vec(path.alphas),
        active: path.active,
        coefficients,
        n_iter: path.n_iter,
    })
}

/// Compute a dense single-output LARS or LARS-Lasso path from sufficient
/// statistics.
///
/// This is the Rust analogue of `sklearn.linear_model.lars_path_gram`: `xy`
/// is `X.T @ y`, `gram` is `X.T @ X`, and `n_samples` is the equivalent sample
/// count used to scale returned alphas and `alpha_min`.
///
/// # Errors
///
/// Returns [`FerroError`] for inconsistent `xy` / `gram` shapes, empty input,
/// non-finite values, invalid `n_samples`, invalid `alpha_min`, non-positive
/// definite Gram matrices, or other numerical failures in the path solver.
pub fn lars_path_gram<F>(
    xy: &Array1<F>,
    gram: &Array2<F>,
    n_samples: usize,
    options: LarsPathOptions<F>,
) -> Result<LarsPathResult<F>, FerroError>
where
    F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static,
{
    validate_lars_path_gram_inputs(xy, gram, n_samples, options.alpha_min)?;
    let (x, y) = gram_to_surrogate_design(xy, gram)?;
    let path = compute_lars_path(
        &x,
        &y,
        options.max_iter,
        options.method == LarsPathMethod::Lasso,
        options.alpha_min,
        Some(n_samples),
    )?;

    let n_features = xy.len();
    let n_alphas = path.alphas.len();
    let mut coefficients = Array2::<F>::zeros((n_features, n_alphas));
    for (alpha_idx, coef) in path.coefficients.iter().enumerate() {
        coefficients.column_mut(alpha_idx).assign(coef);
    }

    Ok(LarsPathResult {
        alphas: Array1::from_vec(path.alphas),
        active: path.active,
        coefficients,
        n_iter: path.n_iter,
    })
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Centred data: `(x_centred, y_centred, x_mean, y_mean)`.
type CentredData<F> = (Array2<F>, Array1<F>, Option<Array1<F>>, Option<F>);

#[derive(Debug, Clone)]
struct LarsPathComputation<F> {
    alphas: Vec<F>,
    active: Vec<usize>,
    coefficients: Vec<Array1<F>>,
    n_iter: usize,
}

fn validate_lars_path_gram_inputs<F: Float>(
    xy: &Array1<F>,
    gram: &Array2<F>,
    n_samples: usize,
    alpha_min: F,
) -> Result<(), FerroError> {
    let (rows, cols) = gram.dim();
    if rows != cols {
        return Err(FerroError::ShapeMismatch {
            expected: vec![rows, rows],
            actual: vec![rows, cols],
            context: "Gram must be square".into(),
        });
    }
    if rows != xy.len() {
        return Err(FerroError::ShapeMismatch {
            expected: vec![rows],
            actual: vec![xy.len()],
            context: "Xy length must match Gram dimension".into(),
        });
    }
    if rows == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: "lars_path_gram requires at least one feature".into(),
        });
    }
    if n_samples == 0 {
        return Err(FerroError::InvalidParameter {
            name: "n_samples".into(),
            reason: "must be positive".into(),
        });
    }
    if gram.iter().any(|v| !v.is_finite()) {
        return Err(FerroError::InvalidParameter {
            name: "Gram".into(),
            reason: "Input Gram contains NaN or infinity.".into(),
        });
    }
    if xy.iter().any(|v| !v.is_finite()) {
        return Err(FerroError::InvalidParameter {
            name: "Xy".into(),
            reason: "Input Xy contains NaN or infinity.".into(),
        });
    }
    if alpha_min < F::zero() || !alpha_min.is_finite() {
        return Err(FerroError::InvalidParameter {
            name: "alpha_min".into(),
            reason: "must be finite and non-negative".into(),
        });
    }
    Ok(())
}

fn gram_to_surrogate_design<F>(
    xy: &Array1<F>,
    gram: &Array2<F>,
) -> Result<(Array2<F>, Array1<F>), FerroError>
where
    F: Float + 'static,
{
    let n_features = xy.len();
    let mut chol = Array2::<F>::zeros((n_features, n_features));
    let eps = F::epsilon();

    for i in 0..n_features {
        for j in 0..=i {
            let mut sum = gram[[i, j]];
            for k in 0..j {
                sum = sum - chol[[i, k]] * chol[[j, k]];
            }
            if i == j {
                if sum <= eps {
                    return Err(FerroError::NumericalInstability {
                        message: "Gram matrix is not positive definite".into(),
                    });
                }
                chol[[i, j]] = sum.sqrt();
            } else {
                let denom = chol[[j, j]];
                if denom.abs() <= eps {
                    return Err(FerroError::NumericalInstability {
                        message: "Gram matrix has a near-zero Cholesky pivot".into(),
                    });
                }
                chol[[i, j]] = sum / denom;
            }
        }
    }

    let mut y = Array1::<F>::zeros(n_features);
    for i in 0..n_features {
        let mut sum = xy[i];
        for j in 0..i {
            sum = sum - chol[[i, j]] * y[j];
        }
        let diag = chol[[i, i]];
        if diag.abs() <= eps {
            return Err(FerroError::NumericalInstability {
                message: "Gram matrix has a near-zero Cholesky pivot".into(),
            });
        }
        y[i] = sum / diag;
    }

    Ok((chol.t().to_owned(), y))
}

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

        // Non-finite input validation (#2259). sklearn `Lars.fit` ->
        // `self._validate_data(X, y, force_writeable=True, y_numeric=True,
        // multi_output=True)` (`_least_angle.py:1183`) keeps the default
        // `force_all_finite=True`, so `check_array` rejects any NaN or +/-inf in
        // X OR y with a `ValueError` BEFORE the LARS path runs.
        // `.iter().any(|v| !v.is_finite())` rejects both NaN and Inf (bounds-safe,
        // no panic, R-CODE-2). `Lars.fit` takes no `sample_weight`. The finite
        // path is byte-identical (the guard never fires on finite input).
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

        let max_active = self.n_nonzero_coefs.unwrap_or(n_features);
        if max_active > n_features {
            return Err(FerroError::InvalidParameter {
                name: "n_nonzero_coefs".into(),
                reason: format!("cannot exceed number of features ({n_features})"),
            });
        }

        let (x_work, y_work, x_mean, y_mean) = center_data(x, y, self.fit_intercept)?;

        let path = compute_lars_path(&x_work, &y_work, max_active, false, F::zero(), None)?;
        let w = path
            .coefficients
            .last()
            .cloned()
            .unwrap_or_else(|| Array1::<F>::zeros(n_features));
        let intercept = compute_intercept(&x_mean, &y_mean, &w);

        Ok(FittedLars {
            coefficients: w,
            intercept,
        })
    }
}

/// Core LARS path computation, shared by [`Lars`] (`method="lar"`) and
/// [`LassoLars`] (`method="lasso"`).
///
/// Walks the LARS equiangular homotopy path. With `lasso_modification = false`
/// this is plain Least Angle Regression: at each step add the maximally
/// correlated feature, move along the equiangular direction until the next
/// feature joins, and stop after `max_steps` features (`n_nonzero_coefs`).
///
/// With `lasso_modification = true` this is the LARS-Lasso homotopy
/// (`sklearn/linear_model/_least_angle.py` `_lars_path_solver`,
/// `method == "lasso"`, `:635`–`:895`): in addition to the join step it
/// computes, for each active variable, the step length at which its
/// coefficient would cross zero (the Efron §3.3 drop length, `z = -coef /
/// least_squares`, `:817`). If a coefficient crosses zero before the next
/// join, the step is truncated at that length (`gamma_ = z_pos`, `:827`), the
/// variable is dropped from the active set after the coefficient update
/// (`:855`–`:894`) and **no** new variable is added on the next iteration
/// (the `if not drop` guard, `:673`). When `alpha_min > 0` the path stops once
/// the maximum correlation `C / n_samples` drops to `alpha_min`, interpolating
/// the final coefficients at exactly `alpha_min` (`:657`–`:669`).
///
/// Returns the full coefficient path. `x` and `y` are assumed centred.
fn compute_lars_path<F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static>(
    x: &Array2<F>,
    y: &Array1<F>,
    max_steps: usize,
    lasso_modification: bool,
    alpha_min: F,
    alpha_n_samples: Option<usize>,
) -> Result<LarsPathComputation<F>, FerroError> {
    let (n_samples, n_features) = x.dim();
    let n_f = F::from(alpha_n_samples.unwrap_or(n_samples)).unwrap_or_else(F::one);
    let mut beta = Array1::<F>::zeros(n_features);
    // Coefficients at the start of the current iteration (sklearn `prev_coef`).
    let mut prev_beta = Array1::<F>::zeros(n_features);
    let mut mu = Array1::<F>::zeros(n_samples);
    let mut active: Vec<usize> = Vec::with_capacity(max_steps.max(1));
    let mut sign_active: Vec<F> = Vec::with_capacity(max_steps.max(1));
    let mut in_active = vec![false; n_features];

    let eps = F::from(1e-12).unwrap_or_else(F::epsilon);
    // sklearn uses np.finfo(np.float32).eps as the alpha equality tolerance
    // (`equality_tolerance`, `:629`).
    let eq_tol = F::from(f32::EPSILON).unwrap_or_else(F::epsilon);

    // `drop` is true when the previous iteration truncated the step at a
    // zero-crossing and removed a variable; on such an iteration no new
    // variable is added (sklearn `if not drop:` guard, `:673`).
    let mut drop = false;
    // `prev_alpha` = C / n_samples from the previous iteration, used for the
    // alpha_min interpolation (`:664`).
    let mut prev_alpha = F::zero();
    let mut path_alphas = Vec::new();
    let mut path_coefficients = Vec::new();

    let mut step = 0;
    loop {
        // Current correlations c = X^T (y - mu) and the maximum |correlation|.
        let residual = y - &mu;
        let mut corr = Array1::<F>::zeros(n_features);
        for j in 0..n_features {
            corr[j] = x.column(j).dot(&residual);
        }
        let mut c_max = F::zero();
        for j in 0..n_features {
            let ac = corr[j].abs();
            if ac > c_max {
                c_max = ac;
            }
        }

        // alpha = C / n_samples (sklearn `:657`). For LARS-Lasso, stop once
        // alpha drops to alpha_min, interpolating the final coefficients at
        // exactly alpha_min (`:658`–`:669`).
        let mut alpha = c_max / n_f;
        if alpha <= alpha_min + eq_tol {
            if (alpha - alpha_min).abs() > eq_tol && step > 0 {
                let denom = prev_alpha - alpha;
                if denom.abs() > eps {
                    let ss = (prev_alpha - alpha_min) / denom;
                    // beta = prev_beta + ss * (beta - prev_beta).
                    for j in 0..n_features {
                        beta[j] = prev_beta[j] + ss * (beta[j] - prev_beta[j]);
                    }
                }
            }
            alpha = alpha_min;
            path_alphas.push(alpha);
            path_coefficients.push(beta.clone());
            break;
        }

        path_alphas.push(alpha);
        path_coefficients.push(beta.clone());
        if c_max <= eps {
            break;
        }
        if step >= max_steps || active.len() >= n_features {
            break;
        }

        // Add the maximally correlated inactive feature, unless the previous
        // iteration dropped a variable (sklearn `if not drop:`, `:673`).
        if !drop {
            let mut j_star: Option<usize> = None;
            let mut best = F::zero();
            for j in 0..n_features {
                if in_active[j] {
                    continue;
                }
                let ac = corr[j].abs();
                if ac > best {
                    best = ac;
                    j_star = Some(j);
                }
            }
            if let Some(j) = j_star {
                active.push(j);
                sign_active.push(if corr[j] >= F::zero() {
                    F::one()
                } else {
                    -F::one()
                });
                in_active[j] = true;
            } else if active.is_empty() {
                break;
            }
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

        // sklearn's signed equiangular weights: least_squares[idx] =
        // sign_active[idx] * w_a[idx] (it folds the sign back in, `:766`).
        let least_squares: Vec<F> = (0..k_a).map(|idx| sign_active[idx] * w_a[idx]).collect();

        // Equiangular direction in sample space: u_vec = X_A @ w_a.
        let u_vec = x_a.dot(&w_a);

        // Join step: gamma chosen so that one new feature joins the active set:
        //   gamma = min over k not in A of the positive ratios
        //       (C_max - c_k) / (A_A - a_k),  (C_max + c_k) / (A_A + a_k)
        // (sklearn g1/g2, `:808`–`:813`).
        let mut gamma = c_max / a_a; // last-step / full-OLS direction
        if active.len() < n_features {
            let mut min_g = F::infinity();
            for j in 0..n_features {
                if in_active[j] {
                    continue;
                }
                let a_j = x.column(j).dot(&u_vec);
                let cands = [(c_max - corr[j], a_a - a_j), (c_max + corr[j], a_a + a_j)];
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

        // Lasso modification (Efron §3.3): the step length at which an active
        // coefficient crosses zero is z = -beta[j] / least_squares[idx]
        // (sklearn `:817`). If the smallest positive z is below the join step,
        // truncate the step there and mark the variable for dropping
        // (`:819`–`:828`).
        let mut drop_idx: Option<usize> = None;
        drop = false;
        if lasso_modification {
            let mut z_pos = F::infinity();
            for (idx, &j) in active.iter().enumerate() {
                let ls = least_squares[idx];
                if ls.abs() <= eps {
                    continue;
                }
                let z = -beta[j] / ls;
                if z > eps && z < z_pos {
                    z_pos = z;
                    drop_idx = Some(idx);
                }
            }
            if let Some(_idx) = drop_idx {
                if z_pos < gamma {
                    gamma = z_pos;
                    drop = true;
                } else {
                    drop_idx = None;
                }
            }
        }

        // Record the iteration-start coefficients, then update beta and mu.
        prev_beta.assign(&beta);
        for (idx, &j) in active.iter().enumerate() {
            beta[j] = beta[j] + gamma * least_squares[idx];
        }
        mu = mu + &(u_vec * gamma);
        prev_alpha = alpha;

        // Drop the zero-crossing variable from the active set (sklearn
        // `:855`–`:894`): remove it, force its coefficient to exactly zero,
        // and do not add a new variable on the next iteration (`drop` stays
        // true so the `if not drop:` guard is skipped).
        if drop && let Some(idx) = drop_idx {
            let j = active[idx];
            beta[j] = F::zero();
            in_active[j] = false;
            active.remove(idx);
            sign_active.remove(idx);
        }

        step += 1;
        if active.is_empty() && !drop {
            break;
        }
    }

    Ok(LarsPathComputation {
        alphas: path_alphas,
        active,
        coefficients: path_coefficients,
        n_iter: step,
    })
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
    /// Routes through the equiangular LARS-Lasso homotopy path
    /// (`compute_lars_path` with `lasso_modification = true`,
    /// `alpha_min = alpha`), mirroring sklearn `_lars_path_solver` (`method == "lasso"`,
    /// `sklearn/linear_model/_least_angle.py:413+`). At each knot the standard
    /// LARS join step competes with the Efron §3.3 drop step (the length at
    /// which an active coefficient crosses zero); the smaller is taken, and on
    /// a drop the crossing variable leaves the active set (its coefficient set
    /// to exactly zero) before the direction is recomputed. The path stops
    /// when the maximum correlation `C / n_samples` reaches `alpha`,
    /// interpolating the final coefficients at exactly `alpha`. This minimizes
    /// the Lasso objective `(1/(2n))||y - Xw||² + alpha·||w||₁`.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] — sample count mismatch.
    /// - [`FerroError::InsufficientSamples`] — zero samples.
    /// - [`FerroError::InvalidParameter`] — `alpha` is negative, or `X`/`y`
    ///   contain NaN or infinity (sklearn `force_all_finite=True` parity).
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedLassoLars<F>, FerroError> {
        let (_n_samples, _n_features) = validate_input(x, y, "LassoLars")?;

        // Non-finite input validation (#2260). `LassoLars.fit` is a SEPARATE
        // override of `Lars.fit` in sklearn (`_least_angle.py:1698`) and calls
        // `self._validate_data(X, y, force_writeable=True, y_numeric=True)`
        // (`_least_angle.py:1726`) with the default `force_all_finite=True`, so
        // `check_array` rejects any NaN or +/-inf in X OR y with a `ValueError`
        // BEFORE the LARS-Lasso path runs. `.iter().any(|v| !v.is_finite())`
        // rejects both NaN and Inf (bounds-safe, no panic, R-CODE-2).
        // `LassoLars.fit` takes no `sample_weight`. The finite path is
        // byte-identical (the guard never fires on finite input).
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

        if self.alpha < F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "alpha".into(),
                reason: "must be non-negative".into(),
            });
        }

        let (x_work, y_work, x_mean, y_mean) = center_data(x, y, self.fit_intercept)?;

        let path = compute_lars_path(&x_work, &y_work, self.max_iter, true, self.alpha, None)?;
        let w = path
            .coefficients
            .last()
            .cloned()
            .unwrap_or_else(|| Array1::<F>::zeros(x.ncols()));
        let intercept = compute_intercept(&x_mean, &y_mean, &w);

        Ok(FittedLassoLars {
            coefficients: w,
            intercept,
        })
    }
}

fn estimate_lasso_lars_ic_noise_variance<F>(
    x: &Array2<F>,
    y: &Array1<F>,
    fit_intercept: bool,
) -> Result<F, FerroError>
where
    F: Float + Send + Sync + ScalarOperand + FromPrimitive + LinalgFloat + 'static,
{
    let (n_samples, n_features) = x.dim();
    let intercept_df = usize::from(fit_intercept);
    if n_samples <= n_features + intercept_df {
        return Err(FerroError::InvalidParameter {
            name: "noise_variance".into(),
            reason: "cannot estimate noise variance when n_samples <= n_features + fit_intercept; \
                     provide noise_variance explicitly"
                .into(),
        });
    }
    let (coef, _rank, _singular) = crate::linalg::solve_lstsq(x, y)?;
    let pred = x.dot(&coef);
    let rss = y
        .iter()
        .zip(pred.iter())
        .fold(<F as num_traits::Zero>::zero(), |acc, (&yi, &pi)| {
            let r = yi - pi;
            acc + r * r
        });
    let denom =
        F::from(n_samples - n_features - intercept_df).unwrap_or_else(<F as num_traits::One>::one);
    Ok(rss / denom)
}

// ---------------------------------------------------------------------------
// Fit — LassoLarsIC
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + LinalgFloat + 'static>
    Fit<Array2<F>, Array1<F>> for LassoLarsIC<F>
{
    type Fitted = FittedLassoLarsIC<F>;
    type Error = FerroError;

    /// Fit the Lasso-LARS IC model.
    ///
    /// Computes the LARS-Lasso coefficient path, evaluates AIC/BIC with degrees
    /// of freedom equal to the number of non-zero coefficients at each knot, and
    /// selects the minimum-criterion alpha. This mirrors sklearn
    /// `LassoLarsIC.fit` for dense single-output input.
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedLassoLarsIC<F>, FerroError> {
        let (n_samples, n_features) = validate_input(x, y, "LassoLarsIC")?;

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
        if let Some(noise_variance) = self.noise_variance
            && (noise_variance < <F as num_traits::Zero>::zero() || !noise_variance.is_finite())
        {
            return Err(FerroError::InvalidParameter {
                name: "noise_variance".into(),
                reason: "must be finite and non-negative".into(),
            });
        }

        let (x_work, y_work, x_mean, y_mean) = center_data(x, y, self.fit_intercept)?;
        let path = compute_lars_path(
            &x_work,
            &y_work,
            self.max_iter,
            true,
            <F as num_traits::Zero>::zero(),
            None,
        )?;
        if path.coefficients.is_empty() {
            return Err(FerroError::NumericalInstability {
                message: "LassoLarsIC path produced no coefficients".into(),
            });
        }

        let noise_variance = match self.noise_variance {
            Some(v) => v,
            None => estimate_lasso_lars_ic_noise_variance(&x_work, &y_work, self.fit_intercept)?,
        };
        let n_samples_f = F::from(n_samples).unwrap_or_else(<F as num_traits::One>::one);
        let criterion_factor = match self.criterion {
            LassoLarsICCriterion::Aic => F::from(2.0).unwrap_or_else(<F as num_traits::One>::one),
            LassoLarsICCriterion::Bic => n_samples_f.ln(),
        };
        let two_pi =
            F::from(2.0 * std::f64::consts::PI).unwrap_or_else(<F as num_traits::One>::one);
        let eps = F::epsilon();

        let mut criterion_values = Vec::with_capacity(path.coefficients.len());
        for coef in &path.coefficients {
            let pred = x_work.dot(coef);
            let rss = y_work.iter().zip(pred.iter()).fold(
                <F as num_traits::Zero>::zero(),
                |acc, (&yi, &pi)| {
                    let r = yi - pi;
                    acc + r * r
                },
            );
            let degrees_of_freedom = coef.iter().filter(|&&c| c.abs() > eps).count();
            let dof_f = F::from(degrees_of_freedom).unwrap_or_else(<F as num_traits::Zero>::zero);
            let value = n_samples_f * (two_pi * noise_variance).ln()
                + rss / noise_variance
                + criterion_factor * dof_f;
            criterion_values.push(value);
        }

        let mut best_idx = 0usize;
        let mut best_value = criterion_values[0];
        for (idx, &value) in criterion_values.iter().enumerate().skip(1) {
            if value < best_value {
                best_value = value;
                best_idx = idx;
            }
        }

        let coefficients = path
            .coefficients
            .get(best_idx)
            .cloned()
            .unwrap_or_else(|| Array1::<F>::zeros(n_features));
        let intercept = compute_intercept(&x_mean, &y_mean, &coefficients);
        let alpha = path
            .alphas
            .get(best_idx)
            .copied()
            .unwrap_or_else(<F as num_traits::Zero>::zero);

        Ok(FittedLassoLarsIC {
            coefficients,
            intercept,
            alpha,
            alphas: Array1::from_vec(path.alphas),
            criterion: Array1::from_vec(criterion_values),
            noise_variance,
            n_iter: path.n_iter,
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
// Predict / HasCoefficients / Pipeline — FittedLassoLarsIC
// ---------------------------------------------------------------------------

impl<F: Float> FittedLassoLarsIC<F> {
    /// Learned coefficient vector.
    #[must_use]
    pub fn coefficients(&self) -> &Array1<F> {
        &self.coefficients
    }

    /// Learned intercept.
    #[must_use]
    pub fn intercept(&self) -> F {
        self.intercept
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>> for FittedLassoLarsIC<F> {
    type Output = Array1<F>;
    type Error = FerroError;

    /// Predict target values for the given feature matrix.
    ///
    /// Computes `X @ coefficients + intercept`.
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

impl<F: Float + Send + Sync + ScalarOperand + 'static> HasCoefficients<F> for FittedLassoLarsIC<F> {
    fn coefficients(&self) -> &Array1<F> {
        &self.coefficients
    }

    fn intercept(&self) -> F {
        self.intercept
    }
}

impl<F> PipelineEstimator<F> for LassoLarsIC<F>
where
    F: Float + FromPrimitive + LinalgFloat + ScalarOperand + Send + Sync + 'static,
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

impl<F> FittedPipelineEstimator<F> for FittedLassoLarsIC<F>
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
                1.0, 0.1, 0.01, 2.0, 0.2, 0.02, 3.0, 0.3, 0.03, 4.0, 0.4, 0.04, 5.0, 0.5, 0.05,
                6.0, 0.6, 0.06, 7.0, 0.7, 0.07, 8.0, 0.8, 0.08, 9.0, 0.9, 0.09, 10.0, 1.0, 0.10,
            ],
        )
        .unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];

        let fitted = Lars::<f64>::new()
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
        assert!(
            Lars::<f64>::new()
                .with_n_nonzero_coefs(5)
                .fit(&x, &y)
                .is_err()
        );
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

        let fitted = LassoLars::<f64>::new().with_alpha(0.0).fit(&x, &y).unwrap();
        assert_relative_eq!(fitted.coefficients()[0], 2.0, epsilon = 0.1);
    }

    #[test]
    fn test_lasso_lars_sparsity() {
        // With high alpha, most coefficients should be zero.
        let x = Array2::from_shape_vec(
            (10, 3),
            vec![
                1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0, 0.0, 0.0, 5.0, 0.0, 0.0, 6.0,
                0.0, 0.0, 7.0, 0.0, 0.0, 8.0, 0.0, 0.0, 9.0, 0.0, 0.0, 10.0, 0.0, 0.0,
            ],
        )
        .unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];

        let fitted = LassoLars::<f64>::new().with_alpha(5.0).fit(&x, &y).unwrap();
        // Irrelevant features (all-zero) should not enter.
        assert_relative_eq!(fitted.coefficients()[1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(fitted.coefficients()[2], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_lasso_lars_negative_alpha() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        assert!(
            LassoLars::<f64>::new()
                .with_alpha(-1.0)
                .fit(&x, &y)
                .is_err()
        );
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
        let fitted = LassoLars::<f64>::new()
            .with_alpha(0.01)
            .fit(&x, &y)
            .unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_lasso_lars_has_coefficients() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        let fitted = LassoLars::<f64>::new()
            .with_alpha(0.01)
            .fit(&x, &y)
            .unwrap();
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
