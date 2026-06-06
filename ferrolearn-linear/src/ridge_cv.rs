//! Ridge regression with built-in cross-validation for alpha selection.
//!
//! This module provides [`RidgeCV`], which automatically selects the best
//! regularization parameter `alpha` from a candidate list. It mirrors
//! scikit-learn's `RidgeCV` (`sklearn/linear_model/_ridge.py`): by **default**
//! (`cv = None`) it uses efficient leave-one-out Generalized Cross-Validation
//! (`_RidgeGCV`, `_ridge.py:1688`), computing the LOO prediction errors in
//! closed form from a single matrix decomposition reused across all `alphas`;
//! when an explicit fold count `cv = Some(k)` is set it falls back to
//! brute-force k-fold cross-validation (mirroring sklearn's `GridSearchCV`
//! branch, `_ridge.py:2413-2439`). In both cases the final model is refit on
//! the full data with the chosen alpha.
//!
//! ## REQ status
//!
//! Two states only (SHIPPED / NOT-STARTED), per goal.md.
//!
//! See `.design/linear/ridge_cv.md` for the full requirements table.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::RidgeCV;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array1, Array2};
//!
//! let model = RidgeCV::<f64>::new();
//! let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
//! let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0];
//!
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 6);
//! ```

use ferray::linalg::LinalgFloat;
use ferray::{Array as FerrayArray, Ix2};
use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasCoefficients;
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2, Axis, ScalarOperand};
use num_traits::{Float, FromPrimitive};

use crate::Ridge;

/// Ridge regression with built-in cross-validation for alpha selection.
///
/// Selects the regularization strength `alpha` from a candidate grid, then
/// refits on the full training data with the chosen alpha. The selection
/// method mirrors scikit-learn's `RidgeCV`:
///
/// - **`cv = None` (the default)** — efficient leave-one-out Generalized
///   Cross-Validation (`_RidgeGCV`, `_ridge.py:1688`). A single matrix
///   decomposition is reused across all `alphas`; for each candidate the LOO
///   prediction errors are computed in closed form and the alpha minimising
///   the mean squared LOO error is selected.
/// - **`cv = Some(k)`** — brute-force k-fold cross-validation (mirroring
///   sklearn's `GridSearchCV` branch, `_ridge.py:2413-2439`): each candidate
///   alpha is scored by averaging the per-fold MSE.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct RidgeCV<F> {
    /// Candidate regularization strengths to evaluate.
    alphas: Vec<F>,
    /// Cross-validation strategy: `None` selects the default leave-one-out
    /// Generalized Cross-Validation path; `Some(k)` selects brute-force
    /// k-fold cross-validation with `k` folds.
    cv: Option<usize>,
    /// Whether to fit an intercept (bias) term.
    fit_intercept: bool,
}

impl<F: Float + FromPrimitive> RidgeCV<F> {
    /// Create a new `RidgeCV` with default settings.
    ///
    /// Defaults: `alphas = [0.1, 1.0, 10.0]`, `cv = None` (leave-one-out
    /// Generalized Cross-Validation, matching sklearn's default), and
    /// `fit_intercept = true`.
    #[must_use]
    pub fn new() -> Self {
        // `F::from(_)` returns `Option`; fall back to `one` (never hit for
        // f32/f64 literals) rather than unwrap in library code (R-CODE-2).
        let one = <F as num_traits::One>::one();
        let p1 = F::from(0.1).unwrap_or(one);
        let ten = F::from(10.0).unwrap_or(one);
        Self {
            alphas: vec![p1, one, ten],
            cv: None,
            fit_intercept: true,
        }
    }

    /// Set the candidate regularization strengths.
    ///
    /// Each value must be non-negative.
    #[must_use]
    pub fn with_alphas(mut self, alphas: Vec<F>) -> Self {
        self.alphas = alphas;
        self
    }

    /// Set the number of cross-validation folds, switching to the brute-force
    /// k-fold path (mirroring sklearn `RidgeCV(cv=k)`).
    ///
    /// Must be at least 2. To use the default leave-one-out Generalized
    /// Cross-Validation path, do not call this (leave `cv = None`).
    #[must_use]
    pub fn with_cv(mut self, cv: usize) -> Self {
        self.cv = Some(cv);
        self
    }

    /// Set whether to fit an intercept term.
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }
}

impl<F: Float + FromPrimitive> Default for RidgeCV<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Ridge regression model with cross-validated alpha.
///
/// Stores the selected alpha, learned coefficients, and intercept. Implements
/// [`Predict`] and [`HasCoefficients`] for introspection.
#[derive(Debug, Clone)]
pub struct FittedRidgeCV<F> {
    /// The alpha that achieved the lowest CV error.
    best_alpha: F,
    /// Learned coefficient vector (one per feature).
    coefficients: Array1<F>,
    /// Learned intercept (bias) term.
    intercept: F,
}

impl<F: Float> FittedRidgeCV<F> {
    /// Returns the alpha value that was selected by cross-validation.
    #[must_use]
    pub fn best_alpha(&self) -> F {
        self.best_alpha
    }
}

/// Split sample indices into `k` CONTIGUOUS folds, mirroring sklearn
/// `KFold(shuffle=False)` (`sklearn/model_selection/_split.py`, `_BaseKFold`):
/// `fold_sizes = full(n_splits, n_samples // n_splits)`,
/// `fold_sizes[: n_samples % n_splits] += 1`, then yield consecutive index
/// blocks. The first `n_samples % k` folds get one extra element; folds are
/// contiguous ranges, NOT round-robin. The `cv = Some(k)` k-fold path resolves
/// to this splitter because sklearn's `GridSearchCV(estimator, cv=k)`
/// (`_ridge.py:2429`) uses `KFold(n_splits=k, shuffle=False)` for a regressor.
/// (Prior identical fix for LassoCV: #421, commit abf5a14.)
fn kfold_indices(n_samples: usize, k: usize) -> Vec<Vec<usize>> {
    let base = n_samples / k;
    let rem = n_samples % k;
    let mut folds = Vec::with_capacity(k);
    let mut current = 0;
    for f in 0..k {
        let fold_size = base + if f < rem { 1 } else { 0 };
        folds.push((current..current + fold_size).collect());
        current += fold_size;
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

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + LinalgFloat + 'static>
    Fit<Array2<F>, Array1<F>> for RidgeCV<F>
{
    type Fitted = FittedRidgeCV<F>;
    type Error = FerroError;

    /// Fit the `RidgeCV` model.
    ///
    /// Selects `alpha` by leave-one-out Generalized Cross-Validation when
    /// `cv = None` (the default; mirrors sklearn's `_RidgeGCV` path,
    /// `_ridge.py:2382`) or by brute-force k-fold cross-validation when
    /// `cv = Some(k)` (mirrors sklearn's `GridSearchCV` path,
    /// `_ridge.py:2413-2439`), then refits on the full data with the chosen
    /// alpha.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] if `x` and `y` have different numbers
    ///   of samples.
    /// - [`FerroError::InvalidParameter`] if `alphas` is empty or contains
    ///   a negative value, or if `cv = Some(k)` with `k < 2`.
    /// - [`FerroError::InsufficientSamples`] if the number of samples is less
    ///   than the number of folds (k-fold path).
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedRidgeCV<F>, FerroError> {
        let (n_samples, _n_features) = x.dim();

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        if self.alphas.is_empty() {
            return Err(FerroError::InvalidParameter {
                name: "alphas".into(),
                reason: "must contain at least one candidate".into(),
            });
        }

        for &a in &self.alphas {
            // `<F as num_traits::Zero>::zero()`: the `LinalgFloat` bound pulls
            // `ferray::Element` (also defining `zero`) into scope, making a
            // bare `F::zero()` ambiguous. Disambiguate to `num_traits::Zero`.
            if a < <F as num_traits::Zero>::zero() {
                return Err(FerroError::InvalidParameter {
                    name: "alphas".into(),
                    reason: "all alpha values must be non-negative".into(),
                });
            }
        }

        // Route on the CV strategy, mirroring sklearn `_BaseRidgeCV.fit`
        // (`_ridge.py:2382`: `if cv is None: ... estimator = _RidgeGCV(...)`,
        // else brute-force `GridSearchCV`).
        let best_alpha = match self.cv {
            None => self.select_alpha_gcv(x, y)?,
            Some(k) => self.select_alpha_kfold(x, y, k)?,
        };

        // Refit on full data with the best alpha, matching sklearn
        // `self.coef_ = estimator.coef_` (`_ridge.py:2441`); `Ridge`'s
        // centering reproduces sklearn's intercept handling.
        let final_model = Ridge::<F>::new()
            .with_alpha(best_alpha)
            .with_fit_intercept(self.fit_intercept);
        let final_fitted = final_model.fit(x, y)?;

        Ok(FittedRidgeCV {
            best_alpha,
            coefficients: final_fitted.coefficients().clone(),
            intercept: final_fitted.intercept(),
        })
    }
}

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + LinalgFloat + 'static> RidgeCV<F> {
    /// Brute-force k-fold alpha selection (sklearn `GridSearchCV` branch,
    /// `_ridge.py:2413-2439`). For each candidate alpha, average the per-fold
    /// MSE of a `Ridge` refit and keep the minimiser.
    fn select_alpha_kfold(&self, x: &Array2<F>, y: &Array1<F>, k: usize) -> Result<F, FerroError> {
        let n_samples = x.nrows();

        if k < 2 {
            return Err(FerroError::InvalidParameter {
                name: "cv".into(),
                reason: "number of folds must be at least 2".into(),
            });
        }

        if n_samples < k {
            return Err(FerroError::InsufficientSamples {
                required: k,
                actual: n_samples,
                context: "RidgeCV requires at least as many samples as folds".into(),
            });
        }

        let folds = kfold_indices(n_samples, k);

        let mut best_alpha = self.alphas[0];
        let mut best_mse = F::infinity();

        for &alpha in &self.alphas {
            let mut total_mse = <F as num_traits::Zero>::zero();

            for fold_idx in 0..k {
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

                let model = Ridge::<F>::new()
                    .with_alpha(alpha)
                    .with_fit_intercept(self.fit_intercept);

                let fitted = model.fit(&x_train, &y_train)?;
                let preds = fitted.predict(&x_test)?;
                total_mse += mse(&y_test, &preds);
            }

            let avg_mse = total_mse / F::from(k).unwrap_or_else(<F as num_traits::One>::one);

            if avg_mse < best_mse {
                best_mse = avg_mse;
                best_alpha = alpha;
            }
        }

        Ok(best_alpha)
    }

    /// Default leave-one-out Generalized Cross-Validation alpha selection,
    /// mirroring sklearn `_RidgeGCV.fit` (`_ridge.py:2059`).
    ///
    /// Centers `X`/`y` when `fit_intercept` (uniform weights here, so
    /// `sqrt_sw = 1`; sklearn `_preprocess_data`, `_ridge.py:2106`), decomposes
    /// once via the shape-appropriate mode (sklearn `_check_gcv_mode`,
    /// `_ridge.py:1569`: SVD of the design matrix when `n_samples > n_features`,
    /// otherwise the eigendecomposition of the Gram matrix `X·Xᵀ`), then for
    /// each alpha computes the closed-form LOO errors and picks the alpha
    /// minimising the mean squared LOO error (`scoring=None` →
    /// `_score_without_scorer = -squared_errors.mean()`, `_ridge.py:2211`).
    fn select_alpha_gcv(&self, x: &Array2<F>, y: &Array1<F>) -> Result<F, FerroError> {
        let (n_samples, n_features) = x.dim();

        // Center X and y (sklearn `_preprocess_data`, `_ridge.py:2106`); with
        // uniform weights the centered design has zero column means and the
        // square-root sample weights are all 1.
        let (x_c, y_c) = if self.fit_intercept {
            let x_mean = x
                .mean_axis(Axis(0))
                .ok_or_else(|| FerroError::NumericalInstability {
                    message: "RidgeCV GCV: failed to compute column means".into(),
                })?;
            let y_mean = y.mean().ok_or_else(|| FerroError::NumericalInstability {
                message: "RidgeCV GCV: failed to compute target mean".into(),
            })?;
            (x - &x_mean, y - y_mean)
        } else {
            (x.to_owned(), y.to_owned())
        };

        // Per-alpha squared LOO errors → mean → negative score; sklearn keeps
        // the alpha with the maximal score, i.e. the minimal mean squared LOO
        // error (`_ridge.py:2146-2186`).
        let scores = if n_samples > n_features {
            self.gcv_scores_svd(&x_c, &y_c)?
        } else {
            self.gcv_scores_eigen(&x_c, &y_c)?
        };

        let mut best_alpha = self.alphas[0];
        let mut best_mse = F::infinity();
        for (&alpha, &mean_sq_err) in self.alphas.iter().zip(scores.iter()) {
            if mean_sq_err < best_mse {
                best_mse = mean_sq_err;
                best_alpha = alpha;
            }
        }

        Ok(best_alpha)
    }

    /// SVD-mode GCV scores (sklearn `_svd_decompose_design_matrix`
    /// `_ridge.py:2025` + `_solve_svd_design_matrix` `_ridge.py:2039`), used
    /// when `n_samples > n_features` and `X` is dense. Returns the mean squared
    /// LOO error for each candidate alpha (lower is better).
    fn gcv_scores_svd(&self, x_c: &Array2<F>, y_c: &Array1<F>) -> Result<Vec<F>, FerroError> {
        let n_samples = x_c.nrows();
        let one = <F as num_traits::One>::one();

        // Build the (possibly intercept-augmented) design matrix. With uniform
        // weights `sqrt_sw = 1`, so the appended intercept column is all ones
        // (sklearn `_svd_decompose_design_matrix`, `_ridge.py:2032`:
        // `intercept_column = sqrt_sw[:, None]`).
        let n_cols = if self.fit_intercept {
            x_c.ncols() + 1
        } else {
            x_c.ncols()
        };
        let mut design = Array2::<F>::zeros((n_samples, n_cols));
        design.slice_mut(ndarray::s![.., ..x_c.ncols()]).assign(x_c);
        if self.fit_intercept {
            design.column_mut(x_c.ncols()).fill(one);
        }

        // Thin SVD: U is (n_samples, k), singular values length k. sklearn uses
        // `linalg.svd(X, full_matrices=0)` (`_ridge.py:2034`).
        let (u, singvals) = svd_u_s(&design)?;
        let k = singvals.len();

        // singvals_sq and UT_y (sklearn `_ridge.py:2035-2036`).
        let singvals_sq: Vec<F> = (0..k).map(|j| singvals[j] * singvals[j]).collect();
        // UT_y[j] = sum_i U[i,j] * y[i].
        let mut ut_y = vec![<F as num_traits::Zero>::zero(); k];
        for j in 0..k {
            let mut acc = <F as num_traits::Zero>::zero();
            for i in 0..n_samples {
                acc += u[(i, j)] * y_c[i];
            }
            ut_y[j] = acc;
        }

        // Intercept dimension: the column of U most aligned with the
        // normalized sqrt_sw vector (uniform weights → ones/√n). Since the
        // query is a unit vector, the most-aligned column maximises
        // |query·U[:,j]| (sklearn `_find_smallest_angle`, `_ridge.py:1579`).
        let intercept_dim = if self.fit_intercept {
            Some(find_intercept_dim(&u, n_samples, k))
        } else {
            None
        };

        let n_f = F::from(n_samples).unwrap_or(one);
        let mut out = Vec::with_capacity(self.alphas.len());
        for &alpha in &self.alphas {
            let inv_alpha = one / alpha;
            // w_j = (singvals_sq_j + alpha)^-1 - alpha^-1 (sklearn :2045).
            let mut w: Vec<F> = singvals_sq
                .iter()
                .map(|&s2| one / (s2 + alpha) - inv_alpha)
                .collect();
            if let Some(d) = intercept_dim {
                // Cancel regularization for the intercept (sklearn :2051).
                w[d] = -inv_alpha;
            }

            // c = U · diag(w) · UT_y + alpha^-1 · y (sklearn :2052).
            // G_inverse_diag_i = sum_j w_j U[i,j]^2 + alpha^-1 (sklearn :2053).
            let mut sum_sq = <F as num_traits::Zero>::zero();
            for i in 0..n_samples {
                let mut c_i = <F as num_traits::Zero>::zero();
                let mut g_i = <F as num_traits::Zero>::zero();
                for j in 0..k {
                    let uij = u[(i, j)];
                    c_i += uij * (w[j] * ut_y[j]);
                    g_i += w[j] * uij * uij;
                }
                c_i += inv_alpha * y_c[i];
                g_i += inv_alpha;
                // looe_i = c_i / G_inverse_diag_i (sklearn :2149).
                let looe = c_i / g_i;
                sum_sq += looe * looe;
            }
            out.push(sum_sq / n_f);
        }
        Ok(out)
    }

    /// Eigen-mode GCV scores, used when `n_samples <= n_features` (sklearn
    /// `_eigen_decompose_gram` `_ridge.py:1900` then `_solve_eigen_gram`
    /// `_ridge.py:1914`). Returns the mean squared LOO error for each candidate
    /// alpha (lower is better).
    fn gcv_scores_eigen(&self, x_c: &Array2<F>, y_c: &Array1<F>) -> Result<Vec<F>, FerroError> {
        let n_samples = x_c.nrows();
        let one = <F as num_traits::One>::one();

        // Gram matrix K = X·Xᵀ on the centered design (sklearn `_compute_gram`,
        // dense → just `X X^T`, `_ridge.py:1799`).
        let mut k_mat = x_c.dot(&x_c.t());
        if self.fit_intercept {
            // Add outer(sqrt_sw, sqrt_sw): with uniform weights this is the
            // all-ones rank-1 matrix, emulating centering with the intercept
            // eigenvector (sklearn `_eigen_decompose_gram`, `_ridge.py:1909`).
            for i in 0..n_samples {
                for j in 0..n_samples {
                    k_mat[(i, j)] += one;
                }
            }
        }

        // Eigendecomposition K = Q diag(eigvals) Qᵀ (sklearn `linalg.eigh`,
        // `_ridge.py:1910`).
        let (eigvals, q) = eigh_sym(&k_mat)?;
        let m = eigvals.len();

        // QT_y[j] = sum_i Q[i,j] * y[i] (sklearn :1911).
        let mut qt_y = vec![<F as num_traits::Zero>::zero(); m];
        for j in 0..m {
            let mut acc = <F as num_traits::Zero>::zero();
            for i in 0..n_samples {
                acc += q[(i, j)] * y_c[i];
            }
            qt_y[j] = acc;
        }

        // Intercept eigenvector: the column of Q most aligned with the
        // normalized sqrt_sw (uniform → ones/√n) (sklearn :1926-1927).
        let intercept_dim = if self.fit_intercept {
            Some(find_intercept_dim(&q, n_samples, m))
        } else {
            None
        };

        let n_f = F::from(n_samples).unwrap_or(one);
        let mut out = Vec::with_capacity(self.alphas.len());
        for &alpha in &self.alphas {
            // w_j = 1 / (eigvals_j + alpha) (sklearn :1919).
            let mut w: Vec<F> = eigvals.iter().map(|&ev| one / (ev + alpha)).collect();
            if let Some(d) = intercept_dim {
                // Cancel regularization for the intercept (sklearn :1928).
                w[d] = <F as num_traits::Zero>::zero();
            }

            // c = Q · diag(w) · QT_y (sklearn :1930).
            // G_inverse_diag_i = sum_j w_j Q[i,j]^2 (sklearn :1931).
            let mut sum_sq = <F as num_traits::Zero>::zero();
            for i in 0..n_samples {
                let mut c_i = <F as num_traits::Zero>::zero();
                let mut g_i = <F as num_traits::Zero>::zero();
                for j in 0..m {
                    let qij = q[(i, j)];
                    c_i += qij * (w[j] * qt_y[j]);
                    g_i += w[j] * qij * qij;
                }
                // looe_i = c_i / G_inverse_diag_i (sklearn :2149).
                let looe = c_i / g_i;
                sum_sq += looe * looe;
            }
            out.push(sum_sq / n_f);
        }
        Ok(out)
    }
}

/// Find the column index of an orthonormal factor (`U` or `Q`, both
/// `(n_samples, k)`) that is most aligned with the normalized uniform-weight
/// vector `ones/√n`. Mirrors sklearn `_find_smallest_angle` (`_ridge.py:1579`):
/// the query and columns are unit vectors, so the most-aligned column maximises
/// `|query · column|`. With `query = ones/√n` the per-column dot product is
/// proportional to the column sum, so `|column-sum|` is the discriminant.
fn find_intercept_dim<F: Float + std::ops::AddAssign + 'static>(
    u: &Array2<F>,
    n_samples: usize,
    k: usize,
) -> usize {
    let mut best_idx = 0usize;
    let mut best_abs = F::neg_infinity();
    for j in 0..k {
        let mut col_sum = <F as num_traits::Zero>::zero();
        for i in 0..n_samples {
            col_sum += u[(i, j)];
        }
        let a = col_sum.abs();
        if a > best_abs {
            best_abs = a;
            best_idx = j;
        }
    }
    best_idx
}

/// Thin SVD via the ferray substrate, returning `(U, S)` as ndarray types.
///
/// Bridges `ndarray → ferray` for the decomposition and back (R-SUBSTRATE-4),
/// routing through [`ferray::linalg::svd`] (`ferray-linalg/src/decomp/svd.rs`).
fn svd_u_s<F: LinalgFloat>(a: &Array2<F>) -> Result<(Array2<F>, Array1<F>), FerroError> {
    let (rows, cols) = a.dim();
    let flat: Vec<F> = a.iter().copied().collect();
    let fa = FerrayArray::<F, Ix2>::from_vec(Ix2::new([rows, cols]), flat).map_err(|e| {
        FerroError::NumericalInstability {
            message: format!("RidgeCV GCV: failed to build design matrix for SVD: {e}"),
        }
    })?;
    let (u, s, _vt) =
        ferray::linalg::svd(&fa, false).map_err(|e| FerroError::NumericalInstability {
            message: format!("RidgeCV GCV: SVD failed: {e}"),
        })?;
    let u_nd = ferray_to_ndarray2(&u)?;
    let s_nd = ferray_to_ndarray1(&s)?;
    Ok((u_nd, s_nd))
}

/// Symmetric eigendecomposition via the ferray substrate, returning
/// `(eigvals, Q)` as ndarray types (ascending eigenvalues).
///
/// Bridges `ndarray → ferray` and back (R-SUBSTRATE-4), routing through
/// [`ferray::linalg::eigh`] (`ferray-linalg/src/decomp/eigen.rs`).
fn eigh_sym<F: LinalgFloat>(a: &Array2<F>) -> Result<(Array1<F>, Array2<F>), FerroError> {
    let (rows, cols) = a.dim();
    let flat: Vec<F> = a.iter().copied().collect();
    let fa = FerrayArray::<F, Ix2>::from_vec(Ix2::new([rows, cols]), flat).map_err(|e| {
        FerroError::NumericalInstability {
            message: format!("RidgeCV GCV: failed to build Gram matrix for eigh: {e}"),
        }
    })?;
    let (vals, q) = ferray::linalg::eigh(&fa).map_err(|e| FerroError::NumericalInstability {
        message: format!("RidgeCV GCV: eigendecomposition failed: {e}"),
    })?;
    let vals_nd = ferray_to_ndarray1(&vals)?;
    let q_nd = ferray_to_ndarray2(&q)?;
    Ok((vals_nd, q_nd))
}

/// Bridge a ferray 2-D array back to `ndarray::Array2` (R-SUBSTRATE-4).
fn ferray_to_ndarray2<F: LinalgFloat>(a: &FerrayArray<F, Ix2>) -> Result<Array2<F>, FerroError> {
    let shape = a.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let nd = a.clone().into_ndarray();
    let flat: Vec<F> = nd.iter().copied().collect();
    Array2::from_shape_vec((rows, cols), flat).map_err(|e| FerroError::NumericalInstability {
        message: format!("RidgeCV GCV: ferray→ndarray (2-D) bridge failed: {e}"),
    })
}

/// Bridge a ferray 1-D array back to `ndarray::Array1` (R-SUBSTRATE-4).
fn ferray_to_ndarray1<F: LinalgFloat>(
    a: &FerrayArray<F, ferray::Ix1>,
) -> Result<Array1<F>, FerroError> {
    let nd = a.clone().into_ndarray();
    let flat: Vec<F> = nd.iter().copied().collect();
    Ok(Array1::from_vec(flat))
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>> for FittedRidgeCV<F> {
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

impl<F: Float + Send + Sync + ScalarOperand + 'static> HasCoefficients<F> for FittedRidgeCV<F> {
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
    fn test_ridge_cv_default_builder() {
        let m = RidgeCV::<f64>::new();
        assert_eq!(m.alphas.len(), 3);
        // Default mirrors sklearn `RidgeCV(cv=None)` → LOO-GCV path.
        assert_eq!(m.cv, None);
        assert!(m.fit_intercept);
    }

    #[test]
    fn test_ridge_cv_with_cv_sets_kfold() {
        let m = RidgeCV::<f64>::new().with_cv(5);
        assert_eq!(m.cv, Some(5));
    }

    #[test]
    fn test_ridge_cv_custom_alphas() {
        let m = RidgeCV::<f64>::new().with_alphas(vec![0.01, 0.1, 1.0, 10.0, 100.0]);
        assert_eq!(m.alphas.len(), 5);
    }

    #[test]
    fn test_ridge_cv_fit_selects_alpha() {
        let x = Array2::from_shape_vec((20, 1), (1..=20).map(f64::from).collect()).unwrap();
        let y = Array1::from_iter((1..=20).map(|i| 2.0 * f64::from(i) + 1.0));

        let model = RidgeCV::<f64>::new()
            .with_alphas(vec![0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
            .with_cv(5);

        let fitted = model.fit(&x, &y).unwrap();

        // For a clean linear relationship, a small alpha should win.
        assert!(fitted.best_alpha() <= 1.0);
    }

    #[test]
    fn test_ridge_cv_predict() {
        let x = Array2::from_shape_vec((10, 1), (1..=10).map(f64::from).collect()).unwrap();
        let y = Array1::from_iter((1..=10).map(|i| 2.0 * f64::from(i) + 1.0));

        let model = RidgeCV::<f64>::new().with_cv(3);
        let fitted = model.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 10);

        // Predictions should be close to the true values.
        for i in 0..10 {
            assert_relative_eq!(preds[i], y[i], epsilon = 1.0);
        }
    }

    #[test]
    fn test_ridge_cv_has_coefficients() {
        let x = Array2::from_shape_vec((10, 2), (0..20).map(f64::from).collect()).unwrap();
        let y = Array1::from_iter((0..10).map(f64::from));

        let model = RidgeCV::<f64>::new().with_cv(3);
        let fitted = model.fit(&x, &y).unwrap();

        assert_eq!(fitted.coefficients().len(), 2);
    }

    #[test]
    fn test_ridge_cv_empty_alphas_error() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];

        let model = RidgeCV::<f64>::new().with_alphas(vec![]);
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_ridge_cv_negative_alpha_error() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];

        let model = RidgeCV::<f64>::new().with_alphas(vec![1.0, -0.5]);
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_ridge_cv_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0];

        let model = RidgeCV::<f64>::new();
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_ridge_cv_insufficient_samples() {
        let x = Array2::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap();
        let y = array![1.0, 2.0];

        let model = RidgeCV::<f64>::new().with_cv(5);
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_ridge_cv_cv_too_small() {
        let x = Array2::from_shape_vec((10, 1), (1..=10).map(f64::from).collect()).unwrap();
        let y = Array1::from_iter((1..=10).map(f64::from));

        let model = RidgeCV::<f64>::new().with_cv(1);
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_ridge_cv_no_intercept() {
        let x = Array2::from_shape_vec((10, 1), (1..=10).map(f64::from).collect()).unwrap();
        let y = Array1::from_iter((1..=10).map(|i| 2.0 * f64::from(i)));

        let model = RidgeCV::<f64>::new().with_cv(3).with_fit_intercept(false);
        let fitted = model.fit(&x, &y).unwrap();

        // With no intercept and origin-passing data, predictions should be close.
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 10);
    }

    #[test]
    fn test_ridge_cv_predict_feature_mismatch() {
        let x_train = Array2::from_shape_vec((10, 2), (0..20).map(f64::from).collect()).unwrap();
        let y = Array1::from_iter((0..10).map(f64::from));

        let fitted = RidgeCV::<f64>::new().with_cv(3).fit(&x_train, &y).unwrap();

        let x_bad = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let result = fitted.predict(&x_bad);
        assert!(result.is_err());
    }

    #[test]
    fn test_kfold_indices_coverage() {
        let folds = kfold_indices(10, 3);
        assert_eq!(folds.len(), 3);

        // Every index 0..10 should appear exactly once.
        let mut all: Vec<usize> = folds.into_iter().flatten().collect();
        all.sort();
        assert_eq!(all, (0..10).collect::<Vec<_>>());
    }

    /// `kfold_indices` must produce CONTIGUOUS folds matching sklearn
    /// `KFold(shuffle=False)` (`_split.py`, `_BaseKFold`), NOT round-robin.
    /// Oracle: `KFold(n_splits=k).split(arange(n))` test folds (live
    /// sklearn 1.5.2). The first `n % k` folds get one extra element.
    #[test]
    fn kfold_indices_contiguous_matches_sklearn() {
        // n=7, k=3 → first 1 fold gets the extra: [0,1,2],[3,4],[5,6].
        assert_eq!(
            kfold_indices(7, 3),
            vec![vec![0, 1, 2], vec![3, 4], vec![5, 6]]
        );
        // n=8, k=4 → n % k == 0, all equal: [0,1],[2,3],[4,5],[6,7].
        assert_eq!(
            kfold_indices(8, 4),
            vec![vec![0, 1], vec![2, 3], vec![4, 5], vec![6, 7]]
        );
        // n=10, k=3 → first 1 fold gets the extra: [0,1,2,3],[4,5,6],[7,8,9].
        assert_eq!(
            kfold_indices(10, 3),
            vec![vec![0, 1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]
        );
    }

    /// End-to-end `RidgeCV` with explicit `cv = Some(4)` on a fixed fixture
    /// (no RNG), isolating the CONTIGUOUS-fold contract from the scoring
    /// metric. ferrolearn's k-fold path averages per-fold MSE, so the matching
    /// oracle is sklearn under `scoring='neg_mean_squared_error'` (live sklearn
    /// 1.5.2): `RidgeCV(alphas=[0.1,1.0,10.0], cv=4,
    /// scoring='neg_mean_squared_error').fit(X, y)` → `alpha_ = 0.1`,
    /// `coef_ = [1.200451, 0.551174, 0.370867]`, `intercept_ = -0.048887`.
    /// This pins that the contiguous folds (`_ridge.py:2429`
    /// `GridSearchCV(cv=k)` → `KFold(shuffle=False)`) drive the train/test
    /// partition correctly.
    ///
    /// NOTE: sklearn's DEFAULT `GridSearchCV` scoring for a regressor is R²,
    /// under which the same call selects `alpha_ = 1.0`. ferrolearn's MSE
    /// scoring is a SEPARATE divergence from this fold fix (issue #429) and is
    /// out of scope here (R-FIX-1: one divergence per dispatch); it is reported
    /// as a spillover finding.
    #[test]
    fn ridge_cv_explicit_cv4_selects_alpha_matches_sklearn() -> Result<(), FerroError> {
        let x = Array2::from_shape_vec(
            (8, 3),
            vec![
                1.0, 0.0, 2.0, //
                0.0, 1.0, 1.0, //
                2.0, 1.0, 0.0, //
                1.0, 2.0, 1.0, //
                0.0, 0.0, 3.0, //
                3.0, 1.0, 2.0, //
                1.0, 1.0, 1.0, //
                2.0, 0.0, 1.0,
            ],
        )
        .map_err(|e| FerroError::NumericalInstability {
            message: format!("test fixture build failed: {e}"),
        })?;
        let y = array![2.0, 1.0, 3.0, 2.5, 1.0, 5.0, 2.0, 2.5];

        let model = RidgeCV::<f64>::new()
            .with_alphas(vec![0.1, 1.0, 10.0])
            .with_cv(4);
        let fitted = model.fit(&x, &y)?;

        assert_relative_eq!(fitted.best_alpha(), 0.1, epsilon = 1e-9);

        let coef = fitted.coefficients();
        assert_relative_eq!(coef[0], 1.200_451, epsilon = 1e-5);
        assert_relative_eq!(coef[1], 0.551_174, epsilon = 1e-5);
        assert_relative_eq!(coef[2], 0.370_867, epsilon = 1e-5);
        assert_relative_eq!(fitted.intercept(), -0.048_887, epsilon = 1e-5);

        Ok(())
    }
}
