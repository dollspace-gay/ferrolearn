//! ElasticNet regression with built-in cross-validation for alpha and
//! l1_ratio selection.
//!
//! This module provides [`ElasticNetCV`], which automatically selects the
//! best `(alpha, l1_ratio)` pair using k-fold cross-validation. For each
//! candidate `l1_ratio`, an alpha grid is generated (or supplied), and the
//! combination that minimises mean squared error is selected.
//!
//! ## REQ status (per `.design/linear/elastic_net_cv.md`, mirrors `sklearn/linear_model/_coordinate_descent.py` @ 1.5.2)
//!
//! Mirrors `sklearn.linear_model.ElasticNetCV` (`_coordinate_descent.py:2131`): per-l1_ratio
//! alpha path (`alpha_max = max|Xᵀy|/(n·l1_ratio)` centered) + contiguous k-fold CV, MSE
//! selection of `(alpha_, l1_ratio_)`, refit. Selection matches the live oracle exactly.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (per-l1_ratio alpha path) | SHIPPED | `compute_alpha_max_enet` (centered, max\|Xᵀy\|/(n·l1_ratio)) + log-spaced grid; members match sklearn `_alpha_grid` to ULP. |
//! | REQ-2 ((alpha,l1_ratio) CV select + refit) | SHIPPED | `alpha_`/`l1_ratio_` match sklearn exactly after #431/#432 fixes. Consumer: `pub use ElasticNetCV` (boundary API). coef_ residual gated by CD-stopping #412 (≤~1e-4). |
//! | REQ-3 (explicit alphas/l1_ratios grids) | SHIPPED | `with_alphas`/`with_l1_ratios`. |
//! | REQ-4 (predict / fit_intercept / HasCoefficients) | SHIPPED | `Predict`/`HasCoefficients for FittedElasticNetCV`. |
//! | REQ-5 (sklearn contiguous KFold) | SHIPPED | #431 fixed (ca90c48) — was round-robin. |
//! | REQ-6 (default l1_ratio=0.5 matching sklearn) | SHIPPED | #432 fixed (ca90c48) — default was a 7-grid; now `[0.5]` (`:2328`). |
//! | REQ-7 (l1_ratio=0 auto-grid raises) | SHIPPED | #440 fixed — auto-grid l1_ratio=0 → `InvalidParameter`, mirroring `_coordinate_descent.py:140`. |
//! | REQ-8..14 NOT-STARTED | mse_path_ (#433), alphas_/dual_gap_/n_iter_ (#434), eps param (#435), positive (#436), n_jobs/precompute (#437), random_state/selection (#438), ferray substrate (#439). coef exact parity gated by #412. |
//! | REQ-15 (non-finite input rejected) | SHIPPED | `Fit for ElasticNetCV::fit` rejects any NaN/+/-inf in X or y BEFORE the per-l1_ratio alpha grid / k-fold split with `FerroError::InvalidParameter`, mirroring sklearn `LinearModelCV.fit` `_validate_data` (default `force_all_finite=True`, `_coordinate_descent.py:1619`/`:1644`) → `ValueError("Input X contains NaN.")` / `"... contains infinity ..."`. ferrolearn's `Fit::fit` takes only `(x, y)` (no `sample_weight` in the trait surface), so X and y are validated. `.iter().any(|v| !v.is_finite())` catches both NaN and Inf; the finite path is byte-identical (alpha_/l1_ratio_/coef_ unchanged). Verified vs the live sklearn 1.5.2 oracle (R-CHAR-3): NaN/+inf/-inf in X and NaN/inf in y all raise `ValueError`; all-finite `alpha_=0.01`/`l1_ratio_=0.5` unchanged (`tests/divergence_linear_nonfinite_batch5.rs::elastic_net_cv_*`). Non-test consumer: the existing `pub use elastic_net_cv::ElasticNetCV` boundary API. (#2265) |
//!
//! acto-critic: per-l1_ratio alpha grid matches sklearn to ULP; 3 divergences found+fixed
//! (#431 folds, #432 default l1_ratio, #440 l1_ratio=0 validation) — `alpha_`/`l1_ratio_` now
//! match the live oracle exactly; coef residual is the tracked #412. Two states only per R-DEFER-2.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::ElasticNetCV;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{Array1, Array2};
//!
//! let model = ElasticNetCV::<f64>::new();
//! let x = Array2::from_shape_vec((10, 1), (1..=10).map(|i| i as f64).collect()).unwrap();
//! let y = Array1::from_iter((1..=10).map(|i| 2.0 * i as f64 + 1.0));
//!
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 10);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasCoefficients;
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2, Axis, ScalarOperand};
use num_traits::{Float, FromPrimitive};

use crate::ElasticNet;

/// ElasticNet regression with built-in cross-validation for joint
/// `(alpha, l1_ratio)` selection.
///
/// For each candidate `l1_ratio`, the module generates a log-spaced alpha
/// grid (from `alpha_max` down to `alpha_max * 1e-3`) or uses the
/// user-supplied grid, runs k-fold CV, and selects the combination that
/// minimises mean squared error.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct ElasticNetCV<F> {
    /// Candidate L1/L2 mixing ratios.
    l1_ratios: Vec<F>,
    /// Number of alphas to generate per l1_ratio when no explicit grid
    /// is supplied.
    n_alphas: usize,
    /// Number of cross-validation folds.
    cv: usize,
    /// Maximum coordinate descent iterations per ElasticNet fit.
    max_iter: usize,
    /// Convergence tolerance for coordinate descent.
    tol: F,
    /// Whether to fit an intercept (bias) term.
    fit_intercept: bool,
}

impl<F: Float + FromPrimitive> ElasticNetCV<F> {
    /// Create a new `ElasticNetCV` with default settings.
    ///
    /// Defaults (mirroring `sklearn.linear_model.ElasticNetCV.__init__`,
    /// `_coordinate_descent.py:2328`, which fixes a single `l1_ratio=0.5`):
    /// - `l1_ratios = [0.5]`
    /// - `n_alphas = 100`
    /// - `cv = 5`
    /// - `max_iter = 1000`
    /// - `tol = 1e-4`
    /// - `fit_intercept = true`
    ///
    /// Use [`with_l1_ratios`](Self::with_l1_ratios) to search a grid of
    /// mixing ratios.
    #[must_use]
    pub fn new() -> Self {
        // sklearn `ElasticNetCV` defaults `l1_ratio=0.5` (a single value),
        // not a grid (`_coordinate_descent.py:2328`). 0.5 is built as
        // `1 / (1 + 1)` so no fallible float conversion is needed here.
        let half = F::one() / (F::one() + F::one());
        Self {
            l1_ratios: vec![half],
            n_alphas: 100,
            cv: 5,
            max_iter: 1000,
            tol: F::from(1e-4).unwrap(),
            fit_intercept: true,
        }
    }

    /// Set the candidate L1/L2 mixing ratios.
    ///
    /// Each value must be in `[0.0, 1.0]`.
    #[must_use]
    pub fn with_l1_ratios(mut self, l1_ratios: Vec<F>) -> Self {
        self.l1_ratios = l1_ratios;
        self
    }

    /// Set the number of alphas generated per `l1_ratio`.
    #[must_use]
    pub fn with_n_alphas(mut self, n_alphas: usize) -> Self {
        self.n_alphas = n_alphas;
        self
    }

    /// Set the number of cross-validation folds.
    ///
    /// Must be at least 2.
    #[must_use]
    pub fn with_cv(mut self, cv: usize) -> Self {
        self.cv = cv;
        self
    }

    /// Set the maximum number of coordinate descent iterations.
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

impl<F: Float + FromPrimitive> Default for ElasticNetCV<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted ElasticNet model with cross-validated `(alpha, l1_ratio)`.
///
/// Stores the selected hyperparameters, learned coefficients, and
/// intercept.
#[derive(Debug, Clone)]
pub struct FittedElasticNetCV<F> {
    /// The alpha that achieved the lowest CV error.
    best_alpha: F,
    /// The l1_ratio that achieved the lowest CV error.
    best_l1_ratio: F,
    /// Learned coefficient vector (some may be exactly zero).
    coefficients: Array1<F>,
    /// Learned intercept (bias) term.
    intercept: F,
}

impl<F: Float> FittedElasticNetCV<F> {
    /// Returns the alpha value selected by cross-validation.
    #[must_use]
    pub fn best_alpha(&self) -> F {
        self.best_alpha
    }

    /// Returns the l1_ratio selected by cross-validation.
    #[must_use]
    pub fn best_l1_ratio(&self) -> F {
        self.best_l1_ratio
    }
}

/// Split sample indices into `k` contiguous folds, mirroring scikit-learn's
/// non-shuffled `KFold._iter_test_indices` (`sklearn/model_selection/_split.py:521-534`).
///
/// Fold sizes are `n_samples / k`, with the first `n_samples % k` folds
/// receiving one extra sample; folds are sequential index blocks. For
/// `n_samples = 12, k = 3` this yields `[0,1,2,3], [4,5,6,7], [8,9,10,11]`;
/// for `n_samples = 10, k = 3` it yields `[0,1,2,3], [4,5,6], [7,8,9]`.
fn kfold_indices(n_samples: usize, k: usize) -> Vec<Vec<usize>> {
    let base = n_samples / k;
    let remainder = n_samples % k;
    let mut folds: Vec<Vec<usize>> = Vec::with_capacity(k);
    let mut current = 0;
    for fold in 0..k {
        let fold_size = if fold < remainder { base + 1 } else { base };
        let stop = current + fold_size;
        folds.push((current..stop).collect());
        current = stop;
    }
    folds
}

/// Compute mean squared error between two arrays.
fn mse<F: Float + FromPrimitive + 'static>(y_true: &Array1<F>, y_pred: &Array1<F>) -> F {
    let n = F::from(y_true.len()).unwrap();
    let diff = y_true - y_pred;
    diff.dot(&diff) / n
}

/// Gather rows from a 2-D array by index.
fn select_rows<F: Float>(x: &Array2<F>, indices: &[usize]) -> Array2<F> {
    let ncols = x.ncols();
    let mut out = Array2::<F>::zeros((indices.len(), ncols));
    for (out_row, &idx) in indices.iter().enumerate() {
        out.row_mut(out_row).assign(&x.row(idx));
    }
    out
}

/// Gather elements from a 1-D array by index.
fn select_elements<F: Float>(y: &Array1<F>, indices: &[usize]) -> Array1<F> {
    Array1::from_iter(indices.iter().map(|&i| y[i]))
}

/// Compute `alpha_max` for ElasticNet given a specific `l1_ratio`.
///
/// `alpha_max = max(|X^T y_centered|) / (n_samples * l1_ratio)`.
/// When `l1_ratio == 0`, falls back to a large default.
fn compute_alpha_max_enet<F: Float + FromPrimitive + ScalarOperand>(
    x: &Array2<F>,
    y: &Array1<F>,
    l1_ratio: F,
    fit_intercept: bool,
) -> F {
    let n = F::from(x.nrows()).unwrap();

    let y_work = if fit_intercept {
        let y_mean = y.mean().unwrap_or_else(F::zero);
        y - y_mean
    } else {
        y.clone()
    };

    let x_work = if fit_intercept {
        let x_mean = x.mean_axis(Axis(0)).unwrap();
        x - &x_mean
    } else {
        x.clone()
    };

    let xty = x_work.t().dot(&y_work);
    let mut max_abs = F::zero();
    for &v in &xty {
        let abs_v = v.abs();
        if abs_v > max_abs {
            max_abs = abs_v;
        }
    }

    if l1_ratio > F::zero() {
        max_abs / (n * l1_ratio)
    } else {
        // Pure Ridge case — use a reasonable default.
        max_abs / n
    }
}

/// Generate `n` log-spaced values from `high` down to `high * eps_ratio`.
fn logspace<F: Float + FromPrimitive>(high: F, eps_ratio: F, n: usize) -> Vec<F> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![high];
    }

    let log_high = high.ln();
    let log_low = (high * eps_ratio).ln();
    let step = (log_low - log_high) / F::from(n - 1).unwrap();

    (0..n)
        .map(|i| (log_high + step * F::from(i).unwrap()).exp())
        .collect()
}

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static> Fit<Array2<F>, Array1<F>>
    for ElasticNetCV<F>
{
    type Fitted = FittedElasticNetCV<F>;
    type Error = FerroError;

    /// Fit the `ElasticNetCV` model.
    ///
    /// For each candidate `l1_ratio`, generates an alpha grid, runs k-fold
    /// CV for every `(alpha, l1_ratio)` pair, then refits on the full data
    /// using the best combination.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] if `x` and `y` sizes differ.
    /// - [`FerroError::InvalidParameter`] if `l1_ratios` is empty, any ratio
    ///   is outside `[0, 1]`, `cv < 2`, or `n_alphas == 0`.
    /// - [`FerroError::InsufficientSamples`] if `n_samples < cv`.
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedElasticNetCV<F>, FerroError> {
        let (n_samples, _n_features) = x.dim();

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        if self.l1_ratios.is_empty() {
            return Err(FerroError::InvalidParameter {
                name: "l1_ratios".into(),
                reason: "must contain at least one candidate".into(),
            });
        }

        for &r in &self.l1_ratios {
            if r < F::zero() || r > F::one() {
                return Err(FerroError::InvalidParameter {
                    name: "l1_ratios".into(),
                    reason: "all l1_ratio values must be in [0, 1]".into(),
                });
            }
        }

        if self.cv < 2 {
            return Err(FerroError::InvalidParameter {
                name: "cv".into(),
                reason: "number of folds must be at least 2".into(),
            });
        }

        if n_samples < self.cv {
            return Err(FerroError::InsufficientSamples {
                required: self.cv,
                actual: n_samples,
                context: "ElasticNetCV requires at least as many samples as folds".into(),
            });
        }

        if self.n_alphas == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_alphas".into(),
                reason: "must be at least 1".into(),
            });
        }

        // Non-finite input validation (#2265 batch5). sklearn `ElasticNetCV.fit`
        // (via `LinearModelCV.fit`) calls `self._validate_data(X, y, ...)`
        // (`_coordinate_descent.py:1619`/`:1644`) — `check_X_params`/
        // `check_y_params` do NOT set `force_all_finite=False`, so the default
        // `True` applies and any NaN or +/-inf in X OR y raises a `ValueError`
        // BEFORE the per-l1_ratio alpha grid / k-fold split. Checking up-front
        // here gives the clean sklearn error before the grid / fold loop.
        // ferrolearn's `Fit::fit` takes only `(x, y)` (no `sample_weight` in the
        // trait surface), so X and y are the validated inputs. `.iter().any(|v|
        // !v.is_finite())` rejects both NaN and Inf (bounds-safe, no panic,
        // R-CODE-2), matching the crate idiom. The finite path is byte-identical
        // (the guard never fires on finite input).
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

        let folds = kfold_indices(n_samples, self.cv);

        let mut best_alpha = F::one();
        let mut best_l1_ratio = self.l1_ratios[0];
        let mut best_mse = F::infinity();

        for &l1_ratio in &self.l1_ratios {
            // Automatic alpha-grid generation is undefined for l1_ratio == 0:
            // alpha_max = max|Xᵀy| / (n * l1_ratio) divides by zero. sklearn's
            // `_alpha_grid` raises ValueError here (`_coordinate_descent.py:140-146`,
            // "Automatic alpha grid generation is not supported for l1_ratio=0").
            // An explicit user-supplied alphas grid would be allowed, but this
            // path always auto-generates, so l1_ratio == 0 is rejected.
            if l1_ratio == F::zero() {
                return Err(FerroError::InvalidParameter {
                    name: "l1_ratio".into(),
                    reason: "Automatic alpha grid generation is not supported for \
                             l1_ratio=0; supply an explicit alphas grid"
                        .into(),
                });
            }

            // Generate alpha grid for this l1_ratio.
            let alpha_max = compute_alpha_max_enet(x, y, l1_ratio, self.fit_intercept);
            // Degenerate branch: when y is constant the centered cross-product is
            // all-zero, so alpha_max == 0. sklearn's `_alpha_grid`
            // (`_coordinate_descent.py:180-183`) tests `alpha_max <=
            // np.finfo(float).resolution` and fills the whole grid with that same
            // resolution. `np.finfo(float)` is ALWAYS np.float64 regardless of the
            // input dtype, so the resolution is the constant 1e-15 for both f32 and
            // f64 (verified live: float32 constant-y input also yields 1e-15, NOT
            // the f32 resolution 1e-6). `unwrap_or_else(F::epsilon)` keeps this
            // panic-free (R-CODE-2); 1e-15 is representable in both f32 and f64.
            let resolution = F::from(1e-15_f64).unwrap_or_else(F::epsilon);
            let alpha_grid = if alpha_max <= resolution {
                vec![resolution; self.n_alphas]
            } else {
                logspace(
                    alpha_max,
                    F::from(1e-3).unwrap_or_else(F::epsilon),
                    self.n_alphas,
                )
            };

            for &alpha in &alpha_grid {
                let mut total_mse = F::zero();

                for fold_idx in 0..self.cv {
                    let test_indices = &folds[fold_idx];
                    let train_indices: Vec<usize> = folds
                        .iter()
                        .enumerate()
                        .filter(|&(i, _)| i != fold_idx)
                        .flat_map(|(_, v)| v.iter().copied())
                        .collect();

                    let x_train = select_rows(x, &train_indices);
                    let y_train = select_elements(y, &train_indices);
                    let x_test = select_rows(x, test_indices);
                    let y_test = select_elements(y, test_indices);

                    let model = ElasticNet::<F>::new()
                        .with_alpha(alpha)
                        .with_l1_ratio(l1_ratio)
                        .with_max_iter(self.max_iter)
                        .with_tol(self.tol)
                        .with_fit_intercept(self.fit_intercept);

                    let fitted = model.fit(&x_train, &y_train)?;
                    let preds = fitted.predict(&x_test)?;
                    total_mse = total_mse + mse(&y_test, &preds);
                }

                let avg_mse = total_mse / F::from(self.cv).unwrap();

                if avg_mse < best_mse {
                    best_mse = avg_mse;
                    best_alpha = alpha;
                    best_l1_ratio = l1_ratio;
                }
            }
        }

        // Refit on full data with the best hyperparameters.
        let final_model = ElasticNet::<F>::new()
            .with_alpha(best_alpha)
            .with_l1_ratio(best_l1_ratio)
            .with_max_iter(self.max_iter)
            .with_tol(self.tol)
            .with_fit_intercept(self.fit_intercept);
        let final_fitted = final_model.fit(x, y)?;

        Ok(FittedElasticNetCV {
            best_alpha,
            best_l1_ratio,
            coefficients: final_fitted.coefficients().clone(),
            intercept: final_fitted.intercept(),
        })
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>>
    for FittedElasticNetCV<F>
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
    for FittedElasticNetCV<F>
{
    fn coefficients(&self) -> &Array1<F> {
        &self.coefficients
    }

    fn intercept(&self) -> F {
        self.intercept
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_elastic_net_cv_default_builder() {
        let m = ElasticNetCV::<f64>::new();
        // sklearn `ElasticNetCV()` defaults to a single `l1_ratio=0.5`
        // (`_coordinate_descent.py:2328`), not a 7-element grid.
        assert_eq!(m.l1_ratios.len(), 1);
        assert_eq!(m.l1_ratios[0], 0.5);
        assert_eq!(m.n_alphas, 100);
        assert_eq!(m.cv, 5);
        assert_eq!(m.max_iter, 1000);
        assert!(m.fit_intercept);
    }

    #[test]
    fn test_elastic_net_cv_builder_setters() {
        let m = ElasticNetCV::<f64>::new()
            .with_l1_ratios(vec![0.5, 0.9])
            .with_n_alphas(20)
            .with_cv(3)
            .with_max_iter(500)
            .with_tol(1e-6)
            .with_fit_intercept(false);
        assert_eq!(m.l1_ratios.len(), 2);
        assert_eq!(m.n_alphas, 20);
        assert_eq!(m.cv, 3);
        assert_eq!(m.max_iter, 500);
        assert!(!m.fit_intercept);
    }

    #[test]
    fn test_elastic_net_cv_fit_selects_params() {
        let x = Array2::from_shape_vec((20, 1), (1..=20).map(f64::from).collect()).unwrap();
        let y = Array1::from_iter((1..=20).map(|i| 2.0 * f64::from(i) + 1.0));

        let model = ElasticNetCV::<f64>::new()
            .with_l1_ratios(vec![0.5, 0.9, 1.0])
            .with_n_alphas(10)
            .with_cv(3);

        let fitted = model.fit(&x, &y).unwrap();

        assert!(fitted.best_alpha() > 0.0);
        assert!(fitted.best_l1_ratio() >= 0.0);
        assert!(fitted.best_l1_ratio() <= 1.0);
    }

    #[test]
    fn test_elastic_net_cv_predict() {
        let x = Array2::from_shape_vec((10, 1), (1..=10).map(f64::from).collect()).unwrap();
        let y = Array1::from_iter((1..=10).map(|i| 2.0 * f64::from(i) + 1.0));

        let model = ElasticNetCV::<f64>::new()
            .with_l1_ratios(vec![0.5, 0.9])
            .with_n_alphas(10)
            .with_cv(3);
        let fitted = model.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 10);

        for i in 0..10 {
            assert_relative_eq!(preds[i], y[i], epsilon = 2.0);
        }
    }

    #[test]
    fn test_elastic_net_cv_has_coefficients() {
        let x = Array2::from_shape_vec((10, 2), (0..20).map(f64::from).collect()).unwrap();
        let y = Array1::from_iter((0..10).map(f64::from));

        let model = ElasticNetCV::<f64>::new()
            .with_l1_ratios(vec![0.5])
            .with_n_alphas(5)
            .with_cv(3);
        let fitted = model.fit(&x, &y).unwrap();

        assert_eq!(fitted.coefficients().len(), 2);
    }

    #[test]
    fn test_elastic_net_cv_empty_l1_ratios_error() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let model = ElasticNetCV::<f64>::new().with_l1_ratios(vec![]);
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_elastic_net_cv_invalid_l1_ratio_error() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let model = ElasticNetCV::<f64>::new().with_l1_ratios(vec![0.5, 1.5]);
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_elastic_net_cv_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0];

        let model = ElasticNetCV::<f64>::new();
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_elastic_net_cv_insufficient_samples() {
        let x = Array2::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap();
        let y = array![1.0, 2.0];

        let model = ElasticNetCV::<f64>::new().with_cv(5);
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_elastic_net_cv_cv_too_small() {
        let x = Array2::from_shape_vec((10, 1), (1..=10).map(f64::from).collect()).unwrap();
        let y = Array1::from_iter((1..=10).map(f64::from));

        let model = ElasticNetCV::<f64>::new().with_cv(1);
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_elastic_net_cv_predict_feature_mismatch() {
        let x_train = Array2::from_shape_vec((10, 2), (0..20).map(f64::from).collect()).unwrap();
        let y = Array1::from_iter((0..10).map(f64::from));

        let fitted = ElasticNetCV::<f64>::new()
            .with_l1_ratios(vec![0.5])
            .with_n_alphas(5)
            .with_cv(3)
            .fit(&x_train, &y)
            .unwrap();

        let x_bad = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let result = fitted.predict(&x_bad);
        assert!(result.is_err());
    }

    #[test]
    fn test_elastic_net_cv_no_intercept() {
        let x = Array2::from_shape_vec((10, 1), (1..=10).map(f64::from).collect()).unwrap();
        let y = Array1::from_iter((1..=10).map(|i| 2.0 * f64::from(i)));

        let model = ElasticNetCV::<f64>::new()
            .with_l1_ratios(vec![0.5])
            .with_n_alphas(5)
            .with_cv(3)
            .with_fit_intercept(false);
        let fitted = model.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 10);
    }

    #[test]
    fn test_elastic_net_cv_l1_ratio_zero_auto_grid_errors() {
        // sklearn's `_alpha_grid` raises ValueError for l1_ratio=0 with no
        // explicit alphas grid ("Automatic alpha grid generation is not
        // supported for l1_ratio=0", `_coordinate_descent.py:140-146`).
        let x = Array2::from_shape_vec((10, 1), (1..=10).map(f64::from).collect()).unwrap();
        let y = Array1::from_iter((1..=10).map(|i| 2.0 * f64::from(i) + 1.0));

        let model = ElasticNetCV::<f64>::new()
            .with_l1_ratios(vec![0.0, 0.5, 1.0])
            .with_n_alphas(5)
            .with_cv(3);
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }
}
