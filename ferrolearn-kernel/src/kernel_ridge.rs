//! Kernel Ridge Regression.
//!
//! [`KernelRidge`] combines Ridge regression with the kernel trick, allowing
//! nonlinear regression via the dual formulation:
//!
//! ```text
//! (K + alpha * I) @ dual_coef = y
//! ```
//!
//! where `K` is the kernel matrix of the training data. Prediction for new
//! data uses:
//!
//! ```text
//! y_pred = K_new @ dual_coef
//! ```
//!
//! This is equivalent to Ridge regression in the kernel-induced feature space
//! but operates directly in the dual (kernel) space, avoiding explicit
//! feature computation.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_kernel::{KernelRidge, KernelType};
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array1, Array2};
//!
//! let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
//! let y = array![0.0, 1.0, 4.0, 9.0, 16.0f64]; // y = x^2
//!
//! let model = KernelRidge::<f64>::new()
//!     .with_alpha(1.0)
//!     .with_kernel(KernelType::Rbf)
//!     .with_gamma(0.5);
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 5);
//! ```
//!
//! Mirrors scikit-learn's `sklearn/kernel_ridge.py` (tag 1.5.2, `KernelRidge`).
//!
//! ## REQ status
//!
//! | REQ | Behavior | Status | Evidence |
//! |-----|----------|--------|----------|
//! | REQ-1 | dual solve `(K + alpha·I) dual_coef = y` | SHIPPED | `fn fit` + `fn cholesky_solve`/`fn gaussian_solve` = sklearn `_solve_cholesky_kernel`; `fn predict` = `K · dual_coef` (`kernel_ridge.py:201-237`) |
//! | REQ-1b | singular-kernel lstsq fallback | SHIPPED | `fn lstsq_min_norm_solve` (eigh-pseudo-inverse, cutoff `n·eps·max\|λ\|`) called by `fn fit` when both `cholesky_solve` and `gaussian_solve` fail = sklearn `_solve_cholesky_kernel` lstsq fallback on `LinAlgError` (`_ridge.py:254-259` `dual_coef = linalg.lstsq(K, y)[0]`); `divergence_singular_kernel_lstsq_fallback{,_min_norm}` match live sklearn min-norm predict `~1e-9`. Warning-absent gap documented (no ferrolearn warning facade). Consumer: same `RsKernelRidge` as REQ-1. |
//! | REQ-2 | linear kernel value parity | SHIPPED | `parity_linear_dual_coef_and_predict` matches live sklearn `~1e-9` |
//! | REQ-3 | rbf kernel value parity (`gamma=None`→`1/n_features` + explicit) | SHIPPED | `parity_rbf_default_and_explicit_gamma` `~1e-9` |
//! | REQ-4 | poly/sigmoid kernel formula (explicit coef0) | SHIPPED | `fn compute_kernel_matrix` (`nystroem.rs`); `parity_poly_explicit_coef0` |
//! | REQ-5 | `coef0` default = 1 | SHIPPED | `KernelRidge::new` sets `coef0 = F::one()` (sklearn `kernel_ridge.py:153`, fixed #1662); `divergence_poly_default_coef0` |
//! | REQ-6 | multi-output y `(n_samples, n_targets)` | NOT-STARTED | `impl Fit<Array2<F>, Array1<F>>` single-output only (`kernel_ridge.py:183`,`:204-212`) — blocker #1665 |
//! | REQ-7 | `sample_weight` | NOT-STARTED | `fn fit` has no sample_weight (`kernel_ridge.py:174`,`:198-199`) — blocker #1665 |
//! | REQ-8 | array-valued `alpha` (per-target) | NOT-STARTED | `alpha: F` scalar (`kernel_ridge.py:202`) — blocker #1665 |
//! | REQ-9 | kernel coverage (laplacian/chi2/additive_chi2/cosine/precomputed/callable/kernel_params) | NOT-STARTED | `KernelType` = Rbf/Polynomial/Linear/Sigmoid only — blocker #1666 |
//! | REQ-10 | parameter validation (alpha≥0, gamma≥0 or None) | SHIPPED | `fn fit` rejects `alpha<0` + `gamma<0` (sklearn `_parameter_constraints` `:134-144`, fixed #1663); `divergence_negative_gamma_not_rejected` |
//! | REQ-11 | sklearn fitted-attr names (`X_fit_`/`n_features_in_`) | SHIPPED | `x_fit()` is the `X_fit_` analog; added `n_features_in()` (= `x_fit.ncols()`, sklearn `kernel_ridge.py:93`); `kernel_ridge_n_features_in_matches_sklearn`. (PyO3 re-export of `X_fit_`/`n_features_in_` rides REQ-substrate/binding.) |
//! | REQ-12 | ferray substrate | NOT-STARTED | `ndarray` + hand-rolled cholesky/gaussian, not `ferray-core`/`ferray::linalg` — blocker #1668 |
//! | REQ-13 | non-test production consumer | SHIPPED | `RsKernelRidge` in `ferrolearn-python` (`extras.rs` + `_extras.py`); param-plumbing gap (only `alpha`) — blocker #1664 |
//!
//! Reference: scikit-learn 1.5.2 (commit 156ef14).

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2};
use num_traits::Float;

use crate::nystroem::{KernelType, compute_kernel_matrix};

/// Kernel Ridge Regression.
///
/// Combines Ridge regression (L2 regularization) with the kernel trick to
/// perform nonlinear regression. The kernel function implicitly maps input
/// features to a high-dimensional space without computing the mapping explicitly.
///
/// # Type Parameters
///
/// - `F`: Float type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct KernelRidge<F> {
    /// Regularization strength (default 1.0).
    alpha: F,
    /// Kernel function to use (default RBF).
    kernel: KernelType,
    /// Kernel parameter for RBF/Sigmoid/Polynomial.
    /// Default: `1.0 / n_features` (set at fit time).
    gamma: Option<F>,
    /// Polynomial degree (default 3).
    degree: usize,
    /// Coefficient for Polynomial/Sigmoid (default 1.0, matching sklearn).
    coef0: F,
}

impl<F: Float + Send + Sync + 'static> KernelRidge<F> {
    /// Create a new `KernelRidge` with default settings.
    ///
    /// Defaults: `alpha = 1.0`, `kernel = Linear`, `gamma = None` (auto),
    /// `degree = 3`, `coef0 = 1.0`.
    ///
    /// The default kernel is `Linear` and the default `coef0` is `1.0` to match
    /// scikit-learn (`KernelRidge(kernel='linear', coef0=1)`).
    #[must_use]
    pub fn new() -> Self {
        Self {
            alpha: F::one(),
            kernel: KernelType::Linear,
            gamma: None,
            degree: 3,
            coef0: F::one(),
        }
    }

    /// Set the regularization strength.
    #[must_use]
    pub fn with_alpha(mut self, alpha: F) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the kernel type.
    #[must_use]
    pub fn with_kernel(mut self, kernel: KernelType) -> Self {
        self.kernel = kernel;
        self
    }

    /// Set the kernel parameter `gamma`.
    #[must_use]
    pub fn with_gamma(mut self, gamma: F) -> Self {
        self.gamma = Some(gamma);
        self
    }

    /// Set the polynomial degree.
    #[must_use]
    pub fn with_degree(mut self, degree: usize) -> Self {
        self.degree = degree;
        self
    }

    /// Set the coefficient for Polynomial/Sigmoid kernels (default 1.0).
    #[must_use]
    pub fn with_coef0(mut self, coef0: F) -> Self {
        self.coef0 = coef0;
        self
    }
}

impl<F: Float + Send + Sync + 'static> Default for KernelRidge<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Kernel Ridge Regression model.
///
/// Stores the training data and dual coefficients. Implements [`Predict`]
/// to generate predictions for new data.
#[derive(Debug, Clone)]
pub struct FittedKernelRidge<F> {
    /// Training feature matrix used during fitting.
    x_fit: Array2<F>,
    /// Dual coefficients: solution to `(K + alpha*I) @ dual_coef = y`.
    dual_coef: Array1<F>,
    /// Kernel type.
    kernel: KernelType,
    /// Effective gamma.
    gamma: F,
    /// Polynomial degree.
    degree: usize,
    /// Coefficient for Polynomial/Sigmoid.
    coef0: F,
}

impl<F: Float + Send + Sync + 'static> FittedKernelRidge<F> {
    /// Return the dual coefficients.
    #[must_use]
    pub fn dual_coef(&self) -> &Array1<F> {
        &self.dual_coef
    }

    /// Return a reference to the stored training data (sklearn `X_fit_`,
    /// `kernel_ridge.py:88`).
    #[must_use]
    pub fn x_fit(&self) -> &Array2<F> {
        &self.x_fit
    }

    /// Number of features seen during fitting (sklearn `n_features_in_`,
    /// `kernel_ridge.py:93`).
    ///
    /// Equals the number of columns of the stored training matrix `X_fit_`.
    #[must_use]
    pub fn n_features_in(&self) -> usize {
        self.x_fit.ncols()
    }

    /// R² coefficient of determination on the given test data.
    /// Equivalent to sklearn's `RegressorMixin.score`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x.nrows() != y.len()` or
    /// the feature count does not match the training data.
    pub fn score(&self, x: &Array2<F>, y: &Array1<F>) -> Result<F, FerroError> {
        if x.nrows() != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows()],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }
        let preds = self.predict(x)?;
        Ok(crate::r2_score(&preds, y))
    }
}

/// Solve symmetric positive-definite system `A @ x = b` via Cholesky decomposition.
///
/// Returns `Err` if the matrix is not positive definite.
fn cholesky_solve<F: Float>(a: &Array2<F>, b: &Array1<F>) -> Result<Array1<F>, FerroError> {
    let n = a.nrows();

    // Compute lower triangular L such that A = L @ L^T
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
                        message: "regularized kernel matrix is not positive definite".into(),
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

/// Symmetric eigendecomposition `A = V diag(w) Vᵀ` via the cyclic Jacobi method.
///
/// `A` must be symmetric (the regularized kernel matrix `K + alpha·I` always is).
/// Returns `(w, V)` where `w` are the eigenvalues and the columns of `V` are the
/// corresponding orthonormal eigenvectors. The method is unconditionally
/// convergent for real symmetric matrices, so it does not error on a singular
/// (rank-deficient) `A` — singular `A` simply yields near-zero eigenvalues,
/// which the caller drops in the pseudo-inverse.
fn jacobi_eigh<F: Float>(a: &Array2<F>) -> (Array1<F>, Array2<F>) {
    let n = a.nrows();
    let mut m = a.clone();
    let mut v = Array2::<F>::eye(n);

    if n <= 1 {
        let mut w = Array1::<F>::zeros(n);
        if n == 1 {
            w[0] = m[[0, 0]];
        }
        return (w, v);
    }

    // Cyclic Jacobi sweeps. 100 sweeps is far beyond the ~log2(n) typically
    // needed; the loop also exits early once the off-diagonal mass is negligible.
    let two = F::one() + F::one();
    for _ in 0..100 {
        // Sum of squared off-diagonal entries (convergence measure).
        let mut off = F::zero();
        for p in 0..n {
            for q in (p + 1)..n {
                off = off + m[[p, q]] * m[[p, q]];
            }
        }
        if off <= F::zero() {
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                let apq = m[[p, q]];
                if apq == F::zero() {
                    continue;
                }
                let app = m[[p, p]];
                let aqq = m[[q, q]];
                // Jacobi rotation angle: cot(2θ) = (aqq - app) / (2·apq).
                let theta = (aqq - app) / (two * apq);
                let t = if theta >= F::zero() {
                    F::one() / (theta + (theta * theta + F::one()).sqrt())
                } else {
                    -F::one() / (-theta + (theta * theta + F::one()).sqrt())
                };
                let c = F::one() / (t * t + F::one()).sqrt();
                let s = t * c;

                // Apply rotation to columns p and q of M, then rows p and q
                // (M := Jᵀ M J), preserving symmetry.
                for k in 0..n {
                    let mkp = m[[k, p]];
                    let mkq = m[[k, q]];
                    m[[k, p]] = c * mkp - s * mkq;
                    m[[k, q]] = s * mkp + c * mkq;
                }
                for k in 0..n {
                    let mpk = m[[p, k]];
                    let mqk = m[[q, k]];
                    m[[p, k]] = c * mpk - s * mqk;
                    m[[q, k]] = s * mpk + c * mqk;
                }
                // Accumulate eigenvectors (V := V J).
                for k in 0..n {
                    let vkp = v[[k, p]];
                    let vkq = v[[k, q]];
                    v[[k, p]] = c * vkp - s * vkq;
                    v[[k, q]] = s * vkp + c * vkq;
                }
            }
        }
    }

    let mut w = Array1::<F>::zeros(n);
    for i in 0..n {
        w[i] = m[[i, i]];
    }
    (w, v)
}

/// Minimum-norm least-squares solution of the symmetric system `A @ x = b`.
///
/// Mirrors scikit-learn's `lstsq` fallback in `_solve_cholesky_kernel`
/// (`sklearn/linear_model/_ridge.py:254-259`): when the direct Cholesky solve
/// raises `LinAlgError` on a singular kernel, sklearn falls back to
/// `scipy.linalg.lstsq(K, y)[0]`, the SVD-based minimum-norm least-squares
/// solution. For a symmetric `A` the SVD min-norm solution equals the
/// eigendecomposition pseudo-inverse `A⁺ b = Σ_{|λ_i|>cutoff} (vᵢᵀb / λ_i) vᵢ`.
///
/// The singular-value cutoff matches `scipy.linalg.lstsq`'s default
/// (`cond=None`): singular values `≤ rcond·max|λ|` are treated as zero, with
/// `rcond = max(M, N)·eps` (the LAPACK gelsd default). Because `A` is square
/// (`M == N == n`), this is `n·eps·max|λ|`.
fn lstsq_min_norm_solve<F: Float + 'static>(a: &Array2<F>, b: &Array1<F>) -> Array1<F> {
    let n = a.nrows();
    if n == 0 {
        return Array1::<F>::zeros(0);
    }

    let (w, v) = jacobi_eigh(a);

    // cutoff = rcond * max|eigenvalue|, rcond = max(M, N) * eps = n * eps.
    let mut max_abs = F::zero();
    for &wi in w.iter() {
        let wabs = wi.abs();
        if wabs > max_abs {
            max_abs = wabs;
        }
    }
    // n as F via repeated addition (F::from is fallible; avoid it).
    let mut n_as_f = F::zero();
    for _ in 0..n {
        n_as_f = n_as_f + F::one();
    }
    let cutoff = n_as_f * F::epsilon() * max_abs;

    // x = V diag(1/λ_i for |λ_i| > cutoff else 0) Vᵀ b.
    let vt_b = v.t().dot(b);
    let mut scaled = Array1::<F>::zeros(n);
    for i in 0..n {
        if w[i].abs() > cutoff {
            scaled[i] = vt_b[i] / w[i];
        }
    }
    v.dot(&scaled)
}

/// Solve `A @ x = b` via Gaussian elimination with partial pivoting (fallback).
fn gaussian_solve<F: Float>(a: &Array2<F>, b: &Array1<F>) -> Result<Array1<F>, FerroError> {
    let n = a.nrows();
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
            let val = aug[[row, col]].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_val < F::from(1e-12).unwrap() {
            return Err(FerroError::NumericalInstability {
                message: "singular matrix in KernelRidge solve".into(),
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

    let mut x = Array1::<F>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = aug[[i, n]];
        for j in (i + 1)..n {
            sum = sum - aug[[i, j]] * x[j];
        }
        if aug[[i, i]].abs() < F::from(1e-12).unwrap() {
            return Err(FerroError::NumericalInstability {
                message: "near-zero pivot in KernelRidge back substitution".into(),
            });
        }
        x[i] = sum / aug[[i, i]];
    }

    Ok(x)
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<F>> for KernelRidge<F> {
    type Fitted = FittedKernelRidge<F>;
    type Error = FerroError;

    /// Fit the Kernel Ridge Regression model.
    ///
    /// Computes the kernel matrix of the training data, adds regularization,
    /// and solves for the dual coefficients.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if `alpha` is negative or if
    /// `gamma` is set to a negative value.
    /// Returns [`FerroError::ShapeMismatch`] if `x` and `y` have different
    /// numbers of samples.
    /// Returns [`FerroError::InsufficientSamples`] if `x` has zero rows.
    /// Returns [`FerroError::NumericalInstability`] if the regularized kernel
    /// matrix is singular.
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedKernelRidge<F>, FerroError> {
        let n_samples = x.nrows();

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "KernelRidge::fit".into(),
            });
        }
        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }
        if self.alpha < F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "alpha".into(),
                reason: "must be non-negative".into(),
            });
        }
        if let Some(g) = self.gamma
            && g < F::zero()
        {
            return Err(FerroError::InvalidParameter {
                name: "gamma".into(),
                reason: "must be non-negative".into(),
            });
        }

        let n_features = x.ncols();
        let effective_gamma = self.gamma.unwrap_or_else(|| {
            if n_features > 0 {
                F::one() / F::from(n_features).unwrap()
            } else {
                F::one()
            }
        });

        // Compute kernel matrix K
        let mut k =
            compute_kernel_matrix(x, x, self.kernel, effective_gamma, self.degree, self.coef0);

        // Add regularization: K + alpha * I
        for i in 0..n_samples {
            k[[i, i]] = k[[i, i]] + self.alpha;
        }

        // Solve (K + alpha*I) @ dual_coef = y.
        //
        // The well-conditioned path is unchanged: a positive-definite kernel is
        // solved by Cholesky (bit-exact with the previous behavior). When both
        // the Cholesky factorization and the Gaussian-elimination fallback fail
        // on a singular kernel (e.g. alpha=0 with duplicate rows), fall back to
        // the minimum-norm least-squares solution, mirroring sklearn's
        // `_solve_cholesky_kernel` lstsq fallback on `LinAlgError`
        // (`sklearn/linear_model/_ridge.py:254-259`:
        // `except np.linalg.LinAlgError: ... dual_coef = linalg.lstsq(K, y)[0]`).
        // sklearn additionally emits a "Singular matrix in solving dual problem"
        // warning; ferrolearn has no warning facade, so the fit simply succeeds.
        let dual_coef = cholesky_solve(&k, y)
            .or_else(|_| gaussian_solve(&k, y))
            .unwrap_or_else(|_| lstsq_min_norm_solve(&k, y));

        Ok(FittedKernelRidge {
            x_fit: x.clone(),
            dual_coef,
            kernel: self.kernel,
            gamma: effective_gamma,
            degree: self.degree,
            coef0: self.coef0,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedKernelRidge<F> {
    type Output = Array1<F>;
    type Error = FerroError;

    /// Predict target values for new data.
    ///
    /// Computes the kernel between new data and training data, then
    /// multiplies by the dual coefficients.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features
    /// does not match the training data.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        if x.ncols() != self.x_fit.ncols() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), self.x_fit.ncols()],
                actual: vec![x.nrows(), x.ncols()],
                context: "KernelRidge::predict feature count must match training data".into(),
            });
        }

        // Compute kernel between new points and training data
        let k_new = compute_kernel_matrix(
            x,
            &self.x_fit,
            self.kernel,
            self.gamma,
            self.degree,
            self.coef0,
        );

        // y_pred = K_new @ dual_coef
        Ok(k_new.dot(&self.dual_coef))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{Array1, Array2, array};

    fn make_data(n: usize, d: usize, seed: u64) -> Array2<f64> {
        use rand::SeedableRng;
        use rand_distr::{Distribution, Normal};
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let data: Vec<f64> = (0..n * d).map(|_| normal.sample(&mut rng)).collect();
        Array2::from_shape_vec((n, d), data).unwrap()
    }

    #[test]
    fn basic_fit_predict() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0.0, 1.0, 4.0, 9.0, 16.0f64];
        let model = KernelRidge::<f64>::new()
            .with_alpha(0.1)
            .with_kernel(KernelType::Rbf)
            .with_gamma(0.5);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 5);
        for &p in preds.iter() {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn kernel_ridge_n_features_in_matches_sklearn() -> Result<(), FerroError> {
        // sklearn KernelRidge().fit(X 3x2,y).n_features_in_ == 2 (= X_fit_.shape[1]).
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0f64]];
        let y = array![1.0, 2.0, 3.0f64];
        let fitted = KernelRidge::<f64>::new().fit(&x, &y)?;
        assert_eq!(fitted.n_features_in(), 2);
        assert_eq!(fitted.n_features_in(), fitted.x_fit().ncols());
        Ok(())
    }

    #[test]
    fn constant_target() {
        let x = Array2::from_shape_vec((10, 1), (0..10).map(|i| i as f64).collect()).unwrap();
        let y = Array1::from_elem(10, 5.0);
        let model = KernelRidge::<f64>::new()
            .with_alpha(0.01)
            .with_kernel(KernelType::Rbf)
            .with_gamma(0.1);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        for &p in preds.iter() {
            assert_abs_diff_eq!(p, 5.0, epsilon = 0.5);
        }
    }

    #[test]
    fn linear_kernel_approximates_linear() {
        // With linear kernel, KRR should approximate linear regression
        // Offset x so linear kernel matrix is non-singular
        let x = Array2::from_shape_vec((10, 1), (1..=10).map(|i| i as f64).collect()).unwrap();
        let y: Array1<f64> = x.column(0).mapv(|xi| 2.0 * xi + 1.0);
        let model = KernelRidge::<f64>::new()
            .with_alpha(0.01)
            .with_kernel(KernelType::Linear);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        for i in 0..10 {
            assert_abs_diff_eq!(preds[i], y[i], epsilon = 1.0);
        }
    }

    #[test]
    fn polynomial_kernel_fits_quadratic() {
        let x = Array2::from_shape_vec((10, 1), (0..10).map(|i| i as f64 * 0.5).collect()).unwrap();
        let y: Array1<f64> = x.column(0).mapv(|xi| xi * xi);
        let model = KernelRidge::<f64>::new()
            .with_alpha(0.01)
            .with_kernel(KernelType::Polynomial)
            .with_gamma(1.0)
            .with_degree(2)
            .with_coef0(0.0);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        for i in 0..10 {
            assert!(
                (preds[i] - y[i]).abs() < 2.0,
                "Poly kernel pred {:.2} vs true {:.2}",
                preds[i],
                y[i]
            );
        }
    }

    #[test]
    fn small_alpha_interpolates() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 4.0, 9.0, 16.0, 25.0f64];
        let model = KernelRidge::<f64>::new()
            .with_alpha(1e-6)
            .with_kernel(KernelType::Rbf)
            .with_gamma(1.0);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        for i in 0..5 {
            assert_abs_diff_eq!(preds[i], y[i], epsilon = 0.1);
        }
    }

    #[test]
    fn large_alpha_shrinks_predictions() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 4.0, 9.0, 16.0, 25.0f64];
        // With very large alpha, dual_coef ≈ y/alpha → predictions near zero
        let model_large = KernelRidge::<f64>::new()
            .with_alpha(1e6)
            .with_kernel(KernelType::Rbf)
            .with_gamma(0.5);
        let fitted_large = model_large.fit(&x, &y).unwrap();
        let preds_large = fitted_large.predict(&x).unwrap();

        let model_small = KernelRidge::<f64>::new()
            .with_alpha(0.01)
            .with_kernel(KernelType::Rbf)
            .with_gamma(0.5);
        let fitted_small = model_small.fit(&x, &y).unwrap();
        let preds_small = fitted_small.predict(&x).unwrap();

        // Large alpha should produce much smaller magnitude predictions
        let mag_large: f64 = preds_large.iter().map(|&p| p.abs()).sum::<f64>() / 5.0;
        let mag_small: f64 = preds_small.iter().map(|&p| p.abs()).sum::<f64>() / 5.0;
        assert!(
            mag_large < mag_small,
            "Large alpha should shrink predictions: large_mag={mag_large:.4}, small_mag={mag_small:.4}"
        );
    }

    #[test]
    fn rejects_mismatched_y_length() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0f64]; // Wrong length
        let model = KernelRidge::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn rejects_negative_alpha() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0f64];
        let model = KernelRidge::<f64>::new().with_alpha(-1.0);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn rejects_empty_input() {
        let x = Array2::<f64>::zeros((0, 3));
        let y = Array1::<f64>::zeros(0);
        let model = KernelRidge::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn predict_rejects_wrong_features() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0f64];
        let fitted = KernelRidge::<f64>::new()
            .with_alpha(1.0)
            .with_kernel(KernelType::Rbf)
            .with_gamma(1.0)
            .fit(&x, &y)
            .unwrap();
        let x_wrong = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]).unwrap();
        assert!(fitted.predict(&x_wrong).is_err());
    }

    #[test]
    fn predict_new_data() {
        let x_train = Array2::from_shape_vec((10, 1), (0..10).map(|i| i as f64).collect()).unwrap();
        let y_train: Array1<f64> = x_train.column(0).mapv(|xi| xi.sin());
        let model = KernelRidge::<f64>::new()
            .with_alpha(0.1)
            .with_kernel(KernelType::Rbf)
            .with_gamma(0.5);
        let fitted = model.fit(&x_train, &y_train).unwrap();

        let x_test = Array2::from_shape_vec((3, 1), vec![0.5, 1.5, 2.5]).unwrap();
        let preds = fitted.predict(&x_test).unwrap();
        assert_eq!(preds.len(), 3);
        for &p in preds.iter() {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn multivariate() {
        let x = make_data(30, 3, 42);
        let y = Array1::from_shape_fn(30, |i| x[[i, 0]] + x[[i, 1]] * x[[i, 2]]);
        let model = KernelRidge::<f64>::new()
            .with_alpha(0.1)
            .with_kernel(KernelType::Rbf)
            .with_gamma(0.5);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 30);
        for &p in preds.iter() {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn f32_support() {
        let data: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let x = Array2::from_shape_vec((10, 1), data).unwrap();
        let y: Array1<f32> = x.column(0).mapv(|xi| xi * xi);
        let model = KernelRidge::<f32>::new()
            .with_alpha(1.0)
            .with_kernel(KernelType::Rbf)
            .with_gamma(0.5);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 10);
        for &p in preds.iter() {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn dual_coef_accessible() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0f64];
        let fitted = KernelRidge::<f64>::new()
            .with_alpha(1.0)
            .with_kernel(KernelType::Rbf)
            .with_gamma(1.0)
            .fit(&x, &y)
            .unwrap();
        assert_eq!(fitted.dual_coef().len(), 5);
        assert_eq!(fitted.x_fit().nrows(), 5);
    }

    #[test]
    fn builder_chain() {
        let model = KernelRidge::<f64>::new()
            .with_alpha(0.5)
            .with_kernel(KernelType::Polynomial)
            .with_gamma(2.0)
            .with_degree(4)
            .with_coef0(1.0);
        assert_eq!(model.degree, 4);
        assert_eq!(model.kernel, KernelType::Polynomial);
    }

    #[test]
    fn zero_alpha_exact_interpolation() {
        // With alpha=0, should exactly interpolate (modulo numerical issues)
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 4.0, 9.0, 16.0, 25.0f64];
        let model = KernelRidge::<f64>::new()
            .with_alpha(0.0)
            .with_kernel(KernelType::Rbf)
            .with_gamma(1.0);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        for i in 0..5 {
            assert_abs_diff_eq!(preds[i], y[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn single_sample() {
        let x = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
        let y = array![5.0f64];
        let model = KernelRidge::<f64>::new()
            .with_alpha(1.0)
            .with_kernel(KernelType::Rbf)
            .with_gamma(1.0);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 1);
        assert!(preds[0].is_finite());
    }
}
