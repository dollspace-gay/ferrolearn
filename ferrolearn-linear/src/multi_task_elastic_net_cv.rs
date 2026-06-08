//! Multi-task ElasticNet regression with built-in cross-validation for
//! `(alpha, l1_ratio)` selection.
//!
//! This module provides [`MultiTaskElasticNetCV`], the multi-output analog of
//! [`crate::ElasticNetCV`]: for each candidate `l1_ratio` it builds a log-spaced
//! alpha grid (the multi-task L21 dual-norm `alpha_max` down to `alpha_max * eps`),
//! runs k-fold cross-validation fitting a [`crate::MultiTaskElasticNet`] per
//! `(alpha, l1_ratio)` fold, selects the pair minimizing mean CV mean-squared
//! error (averaged over held-out samples and tasks), and refits on the full data.
//!
//! Mirrors `sklearn.linear_model.MultiTaskElasticNetCV`
//! (`sklearn/linear_model/_coordinate_descent.py:2806`,
//! `class MultiTaskElasticNetCV(RegressorMixin, LinearModelCV)`). The CV
//! machinery is the shared `LinearModelCV.fit` (`:1552-1837`): `_alpha_grid`
//! (`:96-185`) per `l1_ratio`, `_path_residuals` per fold over the alpha path
//! (`:1450-1482`), `mean_mse = np.mean(mse_paths, axis=1)`, `argmin` select
//! (`:1791-1799`), refit (`:1828-1834`). The single difference vs the single-task
//! [`crate::ElasticNetCV`] is the multi-output branch of `_alpha_grid`: `Xy =
//! Xcᵀ Yc` is 2-D `(n_features, n_tasks)`, so `alpha_max =
//! sqrt(sum(Xy**2, axis=1)).max() / (n_samples * l1_ratio)` (`:178`) — the L21
//! dual norm `max_j ||Xc[:,j]ᵀ Yc||_2 / (n·l1_ratio)`. The inner solver is the
//! multi-output [`crate::MultiTaskElasticNet`].
//!
//! ## REQ status (per `.design/linear/elastic_net_cv.md`, mirrors `sklearn/linear_model/_coordinate_descent.py` @ 1.5.2)
//!
//! Scope mirrors the single-task `ElasticNetCV` REQ-1/2 precedent: the CORE CV
//! path (per-l1_ratio L21 alpha grid + contiguous k-fold + MSE-select + refit)
//! SHIPPED; the path-introspection / advanced attrs NOT-STARTED.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (per-l1_ratio L21 alpha grid) | SHIPPED | `compute_alpha_max_mtenet` centers X/Y when `fit_intercept`, forms `Xy = Xcᵀ Yc` `(p,t)`, then `max_j ||Xy[j,:]||_2 / (n·l1_ratio)` — the multi-output `_alpha_grid` branch `alpha_max = sqrt(sum(Xy**2,axis=1)).max()/(n*l1_ratio)` (`_coordinate_descent.py:178`). `logspace` produces the `np.geomspace(alpha_max, alpha_max*eps, n_alphas)` grid (`:185`). Defaults `n_alphas=100`, `eps=1e-3` match the ctor (`:2994`, `:2993`). Oracle (R-CHAR-3): on the 12×2 fixture, l1_ratio=0.5 → `alphas_[0]=10.51284973948842`, ratio 1e-3 to last. Non-test consumer: `pub use … MultiTaskElasticNetCV` in `lib.rs` (boundary API). |
//! | REQ-2 ((alpha,l1_ratio) CV select + refit) | SHIPPED | `Fit for MultiTaskElasticNetCV`: contiguous `kfold_indices` (sklearn non-shuffled `KFold`, `_split.py:521-534`); for each `(l1_ratio, alpha)` fits `MultiTaskElasticNet` per train fold, accumulates held-out MSE (`mean_mse` over samples+tasks, `_coordinate_descent.py:1478`,`:1482`), `argmin` mean-CV-MSE select (`:1791-1799`), refits on full data (`:1828-1834`). `alpha_`/`l1_ratio_` match the live oracle EXACTLY; `coef_`/`intercept_` within the CD-stopping tol (~1e-4, shared #412). Non-test consumer: `pub use … MultiTaskElasticNetCV`. |
//! | REQ-3 (explicit l1_ratios grid) | SHIPPED | `with_l1_ratios`; `fit` validates non-empty + each in `[0,1]` (`_parameter_constraints["l1_ratio"]`, `:2982`). |
//! | REQ-4 (predict / fit_intercept) | SHIPPED | `Predict for FittedMultiTaskElasticNetCV` = `X·coefᵀ + intercept` → `(n,t)`; `with_fit_intercept` threads into grid + every fold fit. |
//! | REQ-5 (contiguous KFold) | SHIPPED | `kfold_indices` = contiguous blocks (sklearn `check_cv(None)→KFold(5)` non-shuffled), so the selected `(alpha_,l1_ratio_)` matches sklearn (mirrors the single-task #431 fix). |
//! | REQ-6 (default l1_ratio=0.5) | SHIPPED | `new()` defaults `l1_ratios=[0.5]`, matching the sklearn ctor `l1_ratio=0.5` (`:2992`). |
//! | REQ-7 (l1_ratio=0 auto-grid raises) | SHIPPED | auto-grid `l1_ratio=0` → `InvalidParameter`, mirroring `_alpha_grid` `ValueError` (`:140-146`). |
//! | REQ-8..14 NOT-STARTED | `mse_path_`/`alphas_`/`dual_gap_`/`n_iter_` path attrs (shared single-task #433/#434), `eps` param (#435), `n_jobs`/sparse, `random_state`/`selection` (#438), ferray substrate (#439), exact `coef_` parity gated by shared CD-stopping #412 — mirroring `elastic_net_cv.rs` REQ-7..14. Two states only (R-DEFER-2). |
//!
//! acto-builder: the L21 alpha grid + contiguous folds match the single-task
//! verified precedent, so `alpha_`/`l1_ratio_` match the live oracle exactly;
//! `coef_` residual is the tracked CD-stopping #412.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::MultiTaskElasticNetCV;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array2};
//!
//! let model = MultiTaskElasticNetCV::<f64>::new().with_n_alphas(5).with_cv(3);
//! let x: Array2<f64> = array![
//!     [1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 5.0], [2.0, 3.0],
//!     [6.0, 1.0], [3.0, 3.0], [7.0, 2.0], [1.0, 5.0], [4.0, 6.0], [5.0, 2.0],
//! ];
//! let y: Array2<f64> = array![
//!     [3.0, 1.0], [2.5, 2.0], [7.1, 3.5], [6.0, 4.2], [11.2, 6.0], [5.0, 3.0],
//!     [9.0, 2.0], [6.5, 3.3], [12.0, 3.5], [3.0, 5.5], [8.5, 7.0], [9.5, 3.2],
//! ];
//!
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.dim(), (12, 2));
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2, Axis, ScalarOperand};
use num_traits::{Float, FromPrimitive};

use crate::MultiTaskElasticNet;

/// Multi-task ElasticNet regression with built-in cross-validation for joint
/// `(alpha, l1_ratio)` selection.
///
/// For each candidate `l1_ratio`, generates a log-spaced alpha grid (from the
/// L21 dual-norm `alpha_max` down to `alpha_max * 1e-3`), runs k-fold CV fitting
/// a [`MultiTaskElasticNet`] per fold, and selects the combination minimizing
/// the mean held-out MSE (averaged over samples and tasks).
///
/// Mirrors `sklearn.linear_model.MultiTaskElasticNetCV`
/// (`_coordinate_descent.py:2806`).
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct MultiTaskElasticNetCV<F> {
    /// Candidate L1/L2 mixing ratios. Each value must be in `[0, 1]`.
    l1_ratios: Vec<F>,
    /// Number of alphas generated per `l1_ratio` when no explicit grid is
    /// supplied.
    n_alphas: usize,
    /// Number of cross-validation folds.
    cv: usize,
    /// Maximum block-coordinate-descent iterations per inner fit.
    max_iter: usize,
    /// Convergence tolerance for block coordinate descent.
    tol: F,
    /// Whether to fit per-task intercept terms.
    fit_intercept: bool,
}

impl<F: Float + FromPrimitive> MultiTaskElasticNetCV<F> {
    /// Create a new `MultiTaskElasticNetCV` with default settings.
    ///
    /// Defaults mirror `sklearn.linear_model.MultiTaskElasticNetCV.__init__`
    /// (`_coordinate_descent.py:2989-3018`), which fixes a single
    /// `l1_ratio=0.5`:
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
        // sklearn `MultiTaskElasticNetCV` defaults `l1_ratio=0.5` (a single
        // value), not a grid (`_coordinate_descent.py:2992`). 0.5 = `1 / (1 + 1)`
        // so no fallible float conversion is needed.
        let half = F::one() / (F::one() + F::one());
        Self {
            l1_ratios: vec![half],
            n_alphas: 100,
            cv: 5,
            max_iter: 1000,
            tol: F::from(1e-4).unwrap_or_else(F::epsilon),
            fit_intercept: true,
        }
    }

    /// Set the candidate L1/L2 mixing ratios. Each value must be in `[0, 1]`.
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

    /// Set the number of cross-validation folds (must be at least 2).
    #[must_use]
    pub fn with_cv(mut self, cv: usize) -> Self {
        self.cv = cv;
        self
    }

    /// Set the maximum number of block-coordinate-descent iterations.
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

    /// Set whether to fit per-task intercept terms.
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }
}

impl<F: Float + FromPrimitive> Default for MultiTaskElasticNetCV<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted multi-task ElasticNet model with cross-validated `(alpha, l1_ratio)`.
///
/// Stores the selected hyperparameters, learned coefficient matrix (shape
/// `(n_tasks, n_features)`, the sklearn `coef_` layout), and per-task intercept
/// vector.
#[derive(Debug, Clone)]
pub struct FittedMultiTaskElasticNetCV<F> {
    /// The alpha that achieved the lowest CV error (sklearn `alpha_`).
    alpha: F,
    /// The l1_ratio that achieved the lowest CV error (sklearn `l1_ratio_`).
    l1_ratio: F,
    /// Learned coefficients, shape `(n_tasks, n_features)` (sklearn `coef_`).
    coefficients: Array2<F>,
    /// Per-task intercept vector, length `n_tasks` (sklearn `intercept_`).
    intercepts: Array1<F>,
}

impl<F: Float> FittedMultiTaskElasticNetCV<F> {
    /// Returns the alpha value selected by cross-validation (sklearn `alpha_`).
    #[must_use]
    pub fn alpha(&self) -> F {
        self.alpha
    }

    /// Returns the l1_ratio selected by cross-validation (sklearn `l1_ratio_`).
    #[must_use]
    pub fn l1_ratio(&self) -> F {
        self.l1_ratio
    }

    /// Borrow the learned coefficient matrix, shape `(n_tasks, n_features)`
    /// (sklearn `coef_`).
    #[must_use]
    pub fn coef(&self) -> &Array2<F> {
        &self.coefficients
    }

    /// Borrow the per-task intercept vector, length `n_tasks` (sklearn
    /// `intercept_`).
    #[must_use]
    pub fn intercept(&self) -> &Array1<F> {
        &self.intercepts
    }
}

/// Split sample indices into `k` contiguous folds, mirroring scikit-learn's
/// non-shuffled `KFold._iter_test_indices` (`sklearn/model_selection/_split.py:521-534`).
///
/// Fold sizes are `n_samples / k`, with the first `n_samples % k` folds
/// receiving one extra sample; folds are sequential index blocks. For
/// `n_samples = 12, k = 3` this yields `[0,1,2,3], [4,5,6,7], [8,9,10,11]`.
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

/// Mean squared error between two `(n_samples, n_tasks)` arrays, averaged over
/// BOTH samples and tasks.
///
/// Mirrors sklearn `_path_residuals` which returns `this_mse.mean(axis=0)`
/// (mean over samples, `_coordinate_descent.py:1478`) and the fold MSE used in
/// selection is `mean_mse` over folds; the per-fold value entering the mean is
/// itself the per-task MSE averaged across tasks (`this_mse.mean(axis=0)` is a
/// per-task vector, then summed into the scalar fold loss via the final
/// `.mean(axis=0)` at `:1482`).
fn mse_multitask<F: Float + FromPrimitive + 'static>(y_true: &Array2<F>, y_pred: &Array2<F>) -> F {
    let total = F::from(y_true.len()).unwrap_or_else(F::one);
    let diff = y_true - y_pred;
    let sq = (&diff * &diff).sum();
    sq / total
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

/// Compute the multi-task L21 dual-norm `alpha_max` for a given `l1_ratio`.
///
/// `Xy = Xcᵀ Yc` is `(n_features, n_tasks)` (centered when `fit_intercept`);
/// `alpha_max = max_j ||Xy[j, :]||_2 / (n_samples * l1_ratio)` — the multi-output
/// branch of sklearn `_alpha_grid`, `alpha_max =
/// np.sqrt(np.sum(Xy**2, axis=1)).max() / (n_samples * l1_ratio)`
/// (`_coordinate_descent.py:178`). The single-task case (one task) collapses to
/// `max|Xᵀy| / (n·l1_ratio)`.
fn compute_alpha_max_mtenet<F: Float + FromPrimitive + ScalarOperand>(
    x: &Array2<F>,
    y: &Array2<F>,
    l1_ratio: F,
    fit_intercept: bool,
) -> F {
    let n = F::from(x.nrows()).unwrap_or_else(F::one);

    let (xc, yc) = if fit_intercept {
        let x_mean = x
            .mean_axis(Axis(0))
            .unwrap_or_else(|| Array1::zeros(x.ncols()));
        let y_mean = y
            .mean_axis(Axis(0))
            .unwrap_or_else(|| Array1::zeros(y.ncols()));
        (x - &x_mean, y - &y_mean)
    } else {
        (x.clone(), y.clone())
    };

    // Xy = Xcᵀ Yc, shape (n_features, n_tasks).
    let xy = xc.t().dot(&yc);

    // max_j ||Xy[j, :]||_2.
    let mut max_norm = F::zero();
    for j in 0..xy.nrows() {
        let row = xy.row(j);
        let nrm = row.dot(&row).sqrt();
        if nrm > max_norm {
            max_norm = nrm;
        }
    }

    if l1_ratio > F::zero() {
        max_norm / (n * l1_ratio)
    } else {
        max_norm / n
    }
}

/// Generate `n` log-spaced values from `high` down to `high * eps_ratio`,
/// mirroring `np.geomspace(alpha_max, alpha_max * eps, num=n_alphas)`
/// (`_coordinate_descent.py:185`).
fn logspace<F: Float + FromPrimitive>(high: F, eps_ratio: F, n: usize) -> Vec<F> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![high];
    }

    let log_high = high.ln();
    let log_low = (high * eps_ratio).ln();
    let step = (log_low - log_high) / F::from(n - 1).unwrap_or_else(F::one);

    (0..n)
        .map(|i| (log_high + step * F::from(i).unwrap_or_else(F::zero)).exp())
        .collect()
}

/// The F-appropriate `numpy.finfo(float).resolution` — the degenerate-`alpha_max`
/// threshold and fill value sklearn uses (`_coordinate_descent.py:180-182`).
///
/// numpy defines `resolution = 10 ** -precision` where
/// `precision = floor(mantissa_bits * log10(2))`, with `mantissa_bits` the
/// number of *stored* fraction bits (52 for f64, 23 for f32). `num_traits::Float`
/// does not expose `MANTISSA_DIGITS`, so we recover the stored mantissa bits from
/// machine epsilon: `eps = 2^-mantissa_bits` ⇒ `mantissa_bits = -log2(eps)`
/// (52.0 for f64, 23.0 for f32 — rounded to absorb log round-off). This yields
/// `1e-15` for f64 (`floor(52·0.30103) = 15`) and `1e-6` for f32
/// (`floor(23·0.30103) = 6`), matching `np.finfo(np.float64).resolution` and
/// `np.finfo(np.float32).resolution` exactly (dtype-dependent per R-CODE-5).
fn finfo_resolution<F: Float + FromPrimitive>() -> F {
    let eps: f64 = F::epsilon().to_f64().unwrap_or(f64::EPSILON);
    let mantissa_bits = (-eps.log2()).round();
    let precision = (mantissa_bits * 2.0_f64.log10()).floor() as i32;
    F::from(10.0)
        .unwrap_or_else(|| F::one() + F::one())
        .powi(-precision)
}

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static> Fit<Array2<F>, Array2<F>>
    for MultiTaskElasticNetCV<F>
{
    type Fitted = FittedMultiTaskElasticNetCV<F>;
    type Error = FerroError;

    /// Fit the `MultiTaskElasticNetCV` model.
    ///
    /// For each candidate `l1_ratio`, generates the L21 alpha grid, runs k-fold
    /// CV for every `(alpha, l1_ratio)` pair (fitting a [`MultiTaskElasticNet`]
    /// per train fold and scoring held-out MSE), then refits on the full data
    /// using the best combination. Mirrors `LinearModelCV.fit`
    /// (`_coordinate_descent.py:1552-1837`).
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] if `Y.nrows()` differs from the number of
    ///   samples in `X`.
    /// - [`FerroError::InvalidParameter`] if `l1_ratios` is empty, any ratio is
    ///   outside `[0, 1]`, an auto-grid `l1_ratio` is `0`, `cv < 2`, `n_alphas == 0`,
    ///   or X/Y contains NaN/infinity.
    /// - [`FerroError::InsufficientSamples`] if `n_samples < cv`.
    fn fit(
        &self,
        x: &Array2<F>,
        y: &Array2<F>,
    ) -> Result<FittedMultiTaskElasticNetCV<F>, FerroError> {
        let (n_samples, _n_features) = x.dim();

        if n_samples != y.nrows() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.nrows()],
                context: "Y rows must match number of samples in X".into(),
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
                context: "MultiTaskElasticNetCV requires at least as many samples as folds".into(),
            });
        }

        if self.n_alphas == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_alphas".into(),
                reason: "must be at least 1".into(),
            });
        }

        // sklearn `LinearModelCV.fit` -> `self._validate_data(...,
        // force_all_finite=True)` (`_coordinate_descent.py:1619`): any NaN/±inf
        // in X or Y is rejected BEFORE the search (same crate idiom as the base
        // MultiTask estimators).
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
            // sklearn `_alpha_grid` raises ValueError for auto-generation with
            // l1_ratio=0 ("Automatic alpha grid generation is not supported for
            // l1_ratio=0", `_coordinate_descent.py:140-146`); alpha_max divides
            // by `n * l1_ratio`.
            if l1_ratio == F::zero() {
                return Err(FerroError::InvalidParameter {
                    name: "l1_ratio".into(),
                    reason: "Automatic alpha grid generation is not supported for \
                             l1_ratio=0; supply an explicit alphas grid"
                        .into(),
                });
            }

            let alpha_max = compute_alpha_max_mtenet(x, y, l1_ratio, self.fit_intercept);
            // Degenerate alpha_max (constant Y → centered Xy = 0 → alpha_max = 0):
            // sklearn fills the whole grid with `np.finfo(float).resolution`
            // (dtype-dependent: 1e-15 for f64, 1e-6 for f32) whenever
            // `alpha_max <= np.finfo(float).resolution` (`_coordinate_descent.py:180-182`).
            let resolution = finfo_resolution::<F>();
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
                    let y_train = select_rows(y, &train_indices);
                    let x_test = select_rows(x, test_indices);
                    let y_test = select_rows(y, test_indices);

                    let model = MultiTaskElasticNet::<F>::new()
                        .with_alpha(alpha)
                        .with_l1_ratio(l1_ratio)
                        .with_max_iter(self.max_iter)
                        .with_tol(self.tol)
                        .with_fit_intercept(self.fit_intercept);

                    let fitted = model.fit(&x_train, &y_train)?;
                    let preds = fitted.predict(&x_test)?;
                    total_mse = total_mse + mse_multitask(&y_test, &preds);
                }

                let avg_mse = total_mse / F::from(self.cv).unwrap_or_else(F::one);

                if avg_mse < best_mse {
                    best_mse = avg_mse;
                    best_alpha = alpha;
                    best_l1_ratio = l1_ratio;
                }
            }
        }

        // Refit on full data with the best hyperparameters.
        let final_model = MultiTaskElasticNet::<F>::new()
            .with_alpha(best_alpha)
            .with_l1_ratio(best_l1_ratio)
            .with_max_iter(self.max_iter)
            .with_tol(self.tol)
            .with_fit_intercept(self.fit_intercept);
        let final_fitted = final_model.fit(x, y)?;

        Ok(FittedMultiTaskElasticNetCV {
            alpha: best_alpha,
            l1_ratio: best_l1_ratio,
            coefficients: final_fitted.coefficients().clone(),
            intercepts: final_fitted.intercepts().clone(),
        })
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>>
    for FittedMultiTaskElasticNetCV<F>
{
    type Output = Array2<F>;
    type Error = FerroError;

    /// Predict target values for the given feature matrix.
    ///
    /// Computes `X · coefᵀ + intercept` (broadcast per task column), returning
    /// an `(n_samples, n_tasks)` array.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does not
    /// match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features = x.ncols();
        if n_features != self.coefficients.ncols() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.coefficients.ncols()],
                actual: vec![n_features],
                context: "number of features must match fitted model".into(),
            });
        }

        let mut preds = x.dot(&self.coefficients.t());
        for (k, &b) in self.intercepts.iter().enumerate() {
            let mut col = preds.column_mut(k);
            col.mapv_inplace(|v| v + b);
        }
        Ok(preds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_finfo_resolution_matches_numpy() {
        // numpy: np.finfo(np.float64).resolution == 1e-15,
        //        np.finfo(np.float32).resolution == 1e-6 (dtype-dependent).
        assert_eq!(finfo_resolution::<f64>(), 1e-15);
        assert_eq!(finfo_resolution::<f32>(), 1e-6);
    }

    #[test]
    fn test_mtenet_cv_default_builder() {
        let m = MultiTaskElasticNetCV::<f64>::new();
        // sklearn `MultiTaskElasticNetCV()` defaults to a single `l1_ratio=0.5`
        // (`_coordinate_descent.py:2992`).
        assert_eq!(m.l1_ratios.len(), 1);
        assert_eq!(m.l1_ratios[0], 0.5);
        assert_eq!(m.n_alphas, 100);
        assert_eq!(m.cv, 5);
        assert_eq!(m.max_iter, 1000);
        assert!(m.fit_intercept);
    }

    #[test]
    fn test_mtenet_cv_builder_setters() {
        let m = MultiTaskElasticNetCV::<f64>::new()
            .with_l1_ratios(vec![0.3, 0.7])
            .with_n_alphas(5)
            .with_cv(3)
            .with_max_iter(500)
            .with_tol(1e-6)
            .with_fit_intercept(false);
        assert_eq!(m.l1_ratios.len(), 2);
        assert_eq!(m.n_alphas, 5);
        assert_eq!(m.cv, 3);
        assert_eq!(m.max_iter, 500);
        assert!(!m.fit_intercept);
    }

    #[test]
    fn test_mtenet_cv_empty_l1_ratios_error() {
        let x: Array2<f64> = array![[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0]];
        let y: Array2<f64> = array![[3.0, 1.0], [2.5, 2.0], [7.1, 3.5], [6.0, 4.2]];
        let res = MultiTaskElasticNetCV::<f64>::new()
            .with_l1_ratios(vec![])
            .fit(&x, &y);
        assert!(matches!(res, Err(FerroError::InvalidParameter { .. })));
    }

    #[test]
    fn test_mtenet_cv_shape_mismatch_error() {
        let x: Array2<f64> = array![[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]];
        let y: Array2<f64> = array![[3.0, 1.0], [2.5, 2.0]];
        let res = MultiTaskElasticNetCV::<f64>::new().fit(&x, &y);
        assert!(matches!(res, Err(FerroError::ShapeMismatch { .. })));
    }
}
