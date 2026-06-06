//! Principal Component Analysis (PCA).
//!
//! PCA performs linear dimensionality reduction by projecting data onto
//! the directions of maximum variance (principal components). The input
//! data is first centred (mean-subtracted), then the covariance matrix
//! is eigendecomposed to find the top `n_components` directions.
//!
//! # Algorithm
//!
//! 1. Compute the per-feature mean and centre the data.
//! 2. Compute the covariance matrix `C = X_centered^T X_centered / (n - 1)`.
//! 3. Eigendecompose `C` using faer's optimised self-adjoint eigensolver
//!    (for f64/f32), with a Jacobi fallback for other float types.
//! 4. Sort eigenvalues in descending order and retain the top `n_components`.
//! 5. Store the corresponding eigenvectors as rows of `components_`.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_decomp::PCA;
//! use ferrolearn_core::traits::{Fit, Transform};
//! use ndarray::array;
//!
//! let pca = PCA::<f64>::new(1);
//! let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
//! let fitted = pca.fit(&x, &()).unwrap();
//! let projected = fitted.transform(&x).unwrap();
//! assert_eq!(projected.ncols(), 1);
//! ```
//!
//! ## REQ status
//!
//! Design: `.design/decomp/pca.md`. Tracking: #1499. Each REQ is BINARY — SHIPPED
//! (impl + non-test consumer + tests + green verification) or NOT-STARTED (concrete
//! open blocker). Non-test consumers: crate re-export (`lib.rs:98`), the PyO3
//! `_RsPCA` binding (`ferrolearn-python/src/transformers.rs:89`, registered
//! `lib.rs:23`), and `PipelineTransformer` (`pca.rs:509-536`). Oracle = live sklearn
//! 1.5.2 (`_pca.py`), run from `/tmp` (R-CHAR-3). ferrolearn's `fit` exactly mirrors
//! sklearn's `covariance_eigh` solver (`_pca.py:593-644`) including the `svd_flip`
//! sign step.
//!
//! | REQ | Scope | Status | Evidence / Blocker |
//! |---|---|---|---|
//! | REQ-1 | `components_` sign via `svd_flip(u_based_decision=False)` + EXACT value parity (`components_`/`transform`/`explained_variance_`/`ratio`/`singular_values_`) | SHIPPED | `fit` (`pca.rs:461-481`) applies the per-row max-abs-positive flip (numpy `argmax` first-on-ties via strict `>`, whole-row negate) = sklearn `svd_flip` (`_pca.py:647`, `extmath.py:897-905`). `tests/divergence_pca.rs` matches the live sklearn `PCA` oracle element-wise incl. sign to 1e-6 (3 DIV tests, fixtures 6×3/7×4/near-tie). Was #1500, fixed |
//! | REQ-2 | Degenerate/repeated-eigenvalue + rank-deficient component VALUE parity | NOT-STARTED | CARVE-OUT (R-DEFER-3): repeated eigenvalues → faer/LAPACK pick different orthonormal bases (same class as spectral_embedding) — blocker #1501 |
//! | REQ-3 | Components orthonormality (unit rows + mutual orthogonality) | SHIPPED | `test_pca_components_orthonormal` + green-guard; eigenvectors of symmetric covariance |
//! | REQ-4 | `explained_variance_` ordering/non-negativity + `explained_variance_ratio_` (÷ sum of ALL eigenvalues = sklearn `total_var`) | SHIPPED | matches sklearn element-wise to 1e-6 (critic green-guards); `test_pca_explained_variance_*`, `test_pca_n_components_equals_n_features` |
//! | REQ-5 | `singular_values_` = `sqrt(eigval·(n−1))` ≥ 0 | SHIPPED | matches sklearn to 1e-6 (green-guard); `test_pca_singular_values_positive` |
//! | REQ-6 | `inverse_transform` round-trip exact when `n_components == n_features` | SHIPPED | `test_pca_inverse_transform_roundtrip`/`_approx` + green-guard (sign-invariant) |
//! | REQ-7 | Error/parameter contracts (n_components 0/>n_features, n_samples<2, transform/inverse_transform col mismatch) | SHIPPED (scoped) | `fit`/`transform`/`inverse_transform` guards; FLAG: sklearn raises `InvalidParameterError`, accepts `n_components=None` |
//! | REQ-8 | f32 generic support | SHIPPED | `test_pca_f32`; faer f32 eigensolver path |
//! | REQ-9 | `PipelineTransformer` integration | SHIPPED | `pca.rs:509-536`; `test_pca_pipeline_integration` |
//! | REQ-10 | PyO3 `_RsPCA` binding (fit/transform/inverse_transform + components_/explained_variance_/explained_variance_ratio_/mean_/singular_values_ getters) | SHIPPED | `transformers.rs:89`, registered `lib.rs:23`; inherits REQ-1's deterministic signs |
//! | REQ-11 | `whiten` (transform /sqrt(explained_variance_), inverse un-scale) | SHIPPED | `PCA::with_whiten` sets `whiten` (threaded into `FittedPCA`); `Transform::transform` divides each column j by `sqrt(explained_variance_[j])` (eps-clipped) = sklearn `_base.py:157-165`, and `inverse_transform` multiplies back = `_base.py:192-196`. `whiten=false` byte-identical to default. In-module `pca_whiten_transform_matches_sklearn`/`pca_whiten_false_unchanged`/`pca_whiten_inverse_matches_sklearn` match the live sklearn 1.5.2 oracle to 1e-6. Was #1502, fixed |
//! | REQ-12 | `svd_solver` param + full-SVD/randomized/arpack paths | NOT-STARTED | sklearn `_pca.py:519-548,:575-591,:711-778`; ferrolearn covariance_eigh only — blocker #1503 |
//! | REQ-13 | `n_components` as float (variance ratio) / "mle" / None-default | NOT-STARTED | sklearn `_pca.py:657-691`; ferrolearn requires explicit `usize` — blocker #1504 |
//! | REQ-14 | `get_covariance` / `get_precision` | SHIPPED | `FittedPCA::get_covariance` (`pca.rs` symbol `get_covariance`) builds `components_ᵀ·diag(exp_var_diff)·components_ + noise_variance_·I` = sklearn `_base.py:30-56`; `get_precision` (symbol `get_precision`/`precision_and_logdet`) symmetric-eigendecomposes it (same faer `eigen_dispatch` as `fit`) → `V diag(1/λ) Vᵀ` = the unique inverse of sklearn's lemma result (`_base.py:58-101`). In-module `pca_get_covariance_matches_sklearn`/`pca_get_precision_matches_sklearn` match the live sklearn 1.5.2 oracle element-wise to 1e-6. Consumer: `score_samples`/`score` call `precision_and_logdet`; re-export `lib.rs:98`. Was #1505, fixed |
//! | REQ-15 | `score` / `score_samples` (Gaussian log-likelihood) + `noise_variance_` | SHIPPED | `fit` captures the FULL eigenvalue spectrum and sets `noise_variance_ = mean(sorted_eigenvalues[n_comp..min_dim])` = sklearn `_pca.py:685-688` (getter symbol `noise_variance`). `FittedPCA::score_samples` computes `ll_i = −0.5·(Xr_i·precision·Xr_iᵀ) − 0.5·(p·ln(2π) − logdet(precision))` (`logdet = −Σ ln(λ)`) = sklearn `_pca.py:805-830`; `score = mean(score_samples)` = `_pca.py:832-853`. In-module `pca_noise_variance_matches_sklearn`/`pca_score_samples_matches_sklearn`/`pca_score_matches_sklearn` match the live sklearn 1.5.2 oracle to 1e-6 (`whiten=false`). Consumer: `score`/`score_samples` consume `noise_variance_`+`get_precision`; re-export `lib.rs:98`. Was #1507, fixed |
//! | REQ-16 | Fitted attrs `n_components_` / `n_features_in_` | NOT-STARTED | derivable but not exposed — blocker #1508 |
//! | REQ-17 | ctor params `tol`/`iterated_power`/`n_oversamples`/`power_iteration_normalizer`/`random_state`/`copy` | NOT-STARTED | sklearn `_pca.py:407-423`; ferrolearn has `n_components` only — blocker #1509 |
//! | REQ-18 | ferray substrate | NOT-STARTED | dense `ndarray` + direct `faer`/Jacobi — blocker #1510 |
//!
//! Count: **12 SHIPPED (REQ-1,3,4,5,6,7,8,9,10,11,14,15) / 6 NOT-STARTED (REQ-2,12,13,16,17,18)**.

use ferrolearn_core::error::FerroError;
use ferrolearn_core::pipeline::{FittedPipelineTransformer, PipelineTransformer};
use ferrolearn_core::traits::{Fit, Transform};
use ndarray::{Array1, Array2};
use num_traits::Float;
use std::any::TypeId;

// ---------------------------------------------------------------------------
// PCA (unfitted)
// ---------------------------------------------------------------------------

/// Principal Component Analysis configuration.
///
/// Holds the `n_components` hyperparameter. Calling [`Fit::fit`] centres
/// the data, computes the eigendecomposition of the covariance matrix,
/// and returns a [`FittedPCA`] that can project new data.
#[derive(Debug, Clone)]
pub struct PCA<F> {
    /// The number of principal components to retain.
    n_components: usize,
    /// When `true`, [`Transform::transform`] divides each projected component
    /// by `sqrt(explained_variance_)` so the transformed output has unit
    /// component-wise variance, and [`FittedPCA::inverse_transform`]
    /// re-multiplies before reconstructing. Mirrors sklearn `PCA(whiten=...)`
    /// (`sklearn/decomposition/_base.py:157-165` for the transform scale,
    /// `:192-196` for the inverse un-scale). Defaults to `false`.
    whiten: bool,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> PCA<F> {
    /// Create a new `PCA` that retains `n_components` principal components.
    ///
    /// Whitening is disabled by default (sklearn `whiten=False`); enable it
    /// with [`PCA::with_whiten`].
    ///
    /// # Panics
    ///
    /// Does not panic. Validation of `n_components` against the data
    /// dimensions happens during [`Fit::fit`].
    #[must_use]
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            whiten: false,
            _marker: std::marker::PhantomData,
        }
    }

    /// Enable or disable whitening (sklearn `PCA(whiten=...)`).
    ///
    /// When `whiten` is `true`, [`Transform::transform`] divides each projected
    /// component by `sqrt(explained_variance_)`
    /// (`sklearn/decomposition/_base.py:157-165`), and
    /// [`FittedPCA::inverse_transform`] re-multiplies it back (`:192-196`).
    #[must_use]
    pub fn with_whiten(mut self, whiten: bool) -> Self {
        self.whiten = whiten;
        self
    }

    /// Return the number of components this PCA is configured to retain.
    #[must_use]
    pub fn n_components(&self) -> usize {
        self.n_components
    }

    /// Return whether whitening is enabled.
    #[must_use]
    pub fn whiten(&self) -> bool {
        self.whiten
    }
}

// ---------------------------------------------------------------------------
// FittedPCA
// ---------------------------------------------------------------------------

/// A fitted PCA model holding the learned principal components and statistics.
///
/// Created by calling [`Fit::fit`] on a [`PCA`]. Implements
/// [`Transform<Array2<F>>`] to project new data, and provides
/// [`inverse_transform`](FittedPCA::inverse_transform) to reconstruct
/// approximate original data.
#[derive(Debug, Clone)]
pub struct FittedPCA<F> {
    /// Principal component directions, shape `(n_components, n_features)`.
    /// Each row is a unit eigenvector of the covariance matrix.
    components_: Array2<F>,

    /// Variance explained by each component (eigenvalues of the covariance
    /// matrix, sorted descending).
    explained_variance_: Array1<F>,

    /// Ratio of variance explained by each component to total variance.
    explained_variance_ratio_: Array1<F>,

    /// Per-feature mean computed during fitting, used for centring.
    mean_: Array1<F>,

    /// Singular values corresponding to each component.
    singular_values_: Array1<F>,

    /// Estimated noise covariance of the probabilistic-PCA model: the mean of
    /// the DISCARDED (tail) eigenvalues of the sample covariance. Mirrors
    /// sklearn `noise_variance_` (`sklearn/decomposition/_pca.py:686`):
    /// `mean(explained_variance_[n_components:min(n_samples, n_features)])`, or
    /// `0.0` when `n_components >= min(n_samples, n_features)`. Used by
    /// [`FittedPCA::get_covariance`] / [`FittedPCA::get_precision`] /
    /// [`FittedPCA::score_samples`].
    noise_variance_: F,

    /// Whether whitening is applied in `transform`/`inverse_transform`.
    /// Propagated from the unfitted [`PCA::whiten`] setting. Mirrors sklearn
    /// `PCA(whiten=...)` (`sklearn/decomposition/_base.py:157-165,:192-196`).
    whiten: bool,
}

impl<F: Float + Send + Sync + 'static> FittedPCA<F> {
    /// Principal components, shape `(n_components, n_features)`.
    #[must_use]
    pub fn components(&self) -> &Array2<F> {
        &self.components_
    }

    /// Explained variance per component (eigenvalues).
    #[must_use]
    pub fn explained_variance(&self) -> &Array1<F> {
        &self.explained_variance_
    }

    /// Explained variance ratio per component.
    #[must_use]
    pub fn explained_variance_ratio(&self) -> &Array1<F> {
        &self.explained_variance_ratio_
    }

    /// Per-feature mean learned during fitting.
    #[must_use]
    pub fn mean(&self) -> &Array1<F> {
        &self.mean_
    }

    /// Singular values corresponding to each component.
    #[must_use]
    pub fn singular_values(&self) -> &Array1<F> {
        &self.singular_values_
    }

    /// Estimated noise covariance of the probabilistic-PCA model.
    ///
    /// Equals `mean(explained_variance_[n_components:min(n_samples, n_features)])`
    /// — the average of the discarded (tail) eigenvalues of the sample
    /// covariance — or `0` when `n_components >= min(n_samples, n_features)`.
    /// Mirrors sklearn `noise_variance_`
    /// (`sklearn/decomposition/_pca.py:686`).
    #[must_use]
    pub fn noise_variance(&self) -> F {
        self.noise_variance_
    }

    /// Compute the data covariance with the generative (probabilistic-PCA)
    /// model: `cov = components_ᵀ · diag(exp_var_diff) · components_`, then add
    /// `noise_variance_` to the diagonal, where
    /// `exp_var_diff[k] = max(explained_variance_[k] − noise_variance_, 0)`.
    ///
    /// Mirrors sklearn `_BasePCA.get_covariance`
    /// (`sklearn/decomposition/_base.py:30-56`:
    /// `exp_var_diff = where(exp_var > noise_variance_, exp_var − noise_variance_, 0)`;
    /// `cov = (components_.T * exp_var_diff) @ components_`;
    /// `_add_to_diagonal(cov, noise_variance_)`). Returns a
    /// `(n_features, n_features)` matrix.
    ///
    /// # Panics
    ///
    /// Assumes `whiten == false` (the default). When `whiten` is `true` the
    /// returned covariance would need the components un-scaled by
    /// `sqrt(explained_variance_)` (sklearn `_base.py:46-47`); that path is not
    /// yet supported and [`FittedPCA::get_precision`] / score methods return an
    /// error instead.
    #[must_use]
    pub fn get_covariance(&self) -> Array2<F> {
        let n_features = self.mean_.len();
        let n_comp = self.components_.nrows();

        // exp_var_diff[k] = max(explained_variance_[k] - noise_variance_, 0)
        // (sklearn `_base.py:48-53` with the `where(... , 0)` clamp).
        let mut exp_var_diff = Array1::<F>::zeros(n_comp);
        for k in 0..n_comp {
            let diff = self.explained_variance_[k] - self.noise_variance_;
            exp_var_diff[k] = if self.explained_variance_[k] > self.noise_variance_ {
                diff
            } else {
                F::zero()
            };
        }

        // cov = components_ᵀ · diag(exp_var_diff) · components_
        //     = Σ_k exp_var_diff[k] · (component_k ⊗ component_k)
        // (sklearn `cov = (components_.T * exp_var_diff) @ components_`,
        // `_base.py:54`).
        let mut cov = Array2::<F>::zeros((n_features, n_features));
        for k in 0..n_comp {
            let w = exp_var_diff[k];
            for i in 0..n_features {
                let ci = self.components_[[k, i]];
                for j in 0..n_features {
                    cov[[i, j]] = cov[[i, j]] + w * ci * self.components_[[k, j]];
                }
            }
        }

        // cov[i,i] += noise_variance_ (sklearn `_add_to_diagonal`, `_base.py:55`).
        for i in 0..n_features {
            cov[[i, i]] = cov[[i, i]] + self.noise_variance_;
        }

        cov
    }

    /// Compute the data precision matrix (inverse of [`get_covariance`]) of the
    /// generative (probabilistic-PCA) model.
    ///
    /// sklearn computes this via the matrix-inversion lemma for efficiency
    /// (`sklearn/decomposition/_base.py:58-101`); the inverse of a symmetric
    /// positive-definite matrix is unique, so this implementation instead
    /// symmetric-eigendecomposes [`get_covariance`] (`cov = V diag(λ) Vᵀ`) and
    /// returns `precision = V diag(1/λ) Vᵀ`, using the SAME faer
    /// `self_adjoint_eigen` routine as [`Fit::fit`]. This is numerically
    /// equivalent to (and element-wise matches) sklearn's lemma result.
    ///
    /// The eigenvalues `λ` of `get_covariance` are also what
    /// [`score_samples`](FittedPCA::score_samples) needs for
    /// `logdet(precision) = −Σ ln(λ)`.
    ///
    /// [`get_covariance`]: FittedPCA::get_covariance
    ///
    /// # Errors
    ///
    /// - [`FerroError::NumericalInstability`] if `whiten` is enabled (this path
    ///   is not yet supported), if the eigendecomposition fails, or if any
    ///   eigenvalue of the covariance is `<= 0` (the covariance is not positive
    ///   definite, so its inverse is ill-conditioned).
    pub fn get_precision(&self) -> Result<Array2<F>, FerroError> {
        let (precision, _logdet) = self.precision_and_logdet()?;
        Ok(precision)
    }

    /// Compute both the precision matrix and `logdet(precision)` in one
    /// eigendecomposition of [`get_covariance`].
    ///
    /// Returns `(precision, logdet_precision)` where
    /// `precision = V diag(1/λ) Vᵀ` and
    /// `logdet_precision = −Σ ln(λ)` (λ = eigenvalues of `get_covariance`).
    /// Shared by [`get_precision`](FittedPCA::get_precision) and
    /// [`score_samples`](FittedPCA::score_samples) so the score path does not
    /// re-eigendecompose. Cites sklearn `get_precision`
    /// (`_base.py:58-101`) and the `fast_logdet(precision)` term of
    /// `score_samples` (`_pca.py:829`).
    fn precision_and_logdet(&self) -> Result<(Array2<F>, F), FerroError> {
        if self.whiten {
            return Err(FerroError::NumericalInstability {
                message: "score/get_precision with whiten=true not yet supported".into(),
            });
        }

        let cov = self.get_covariance();
        let n_features = cov.nrows();

        // Symmetric eigendecomposition of the PD covariance — the SAME routine
        // `Fit::fit` uses for the covariance matrix (faer `self_adjoint_eigen`,
        // Jacobi fallback). `eigenvectors` is column-major.
        let max_jacobi_iter = n_features * n_features * 100 + 1000;
        let (eigenvalues, eigenvectors) = eigen_dispatch(&cov, max_jacobi_iter)?;

        // precision = V diag(1/λ) Vᵀ ; logdet(precision) = −Σ ln(λ).
        let mut log_det = F::zero();
        let mut inv_eig = Array1::<F>::zeros(n_features);
        for i in 0..n_features {
            let lambda = eigenvalues[i];
            if lambda <= F::zero() {
                return Err(FerroError::NumericalInstability {
                    message: "PCA covariance is not positive definite (eigenvalue <= 0); \
                              cannot compute precision"
                        .into(),
                });
            }
            inv_eig[i] = F::one() / lambda;
            log_det = log_det - lambda.ln();
        }

        // precision[i,j] = Σ_m V[i,m] · (1/λ_m) · V[j,m]
        let mut precision = Array2::<F>::zeros((n_features, n_features));
        for i in 0..n_features {
            for j in 0..n_features {
                let mut acc = F::zero();
                for m in 0..n_features {
                    acc = acc + eigenvectors[[i, m]] * inv_eig[m] * eigenvectors[[j, m]];
                }
                precision[[i, j]] = acc;
            }
        }

        Ok((precision, log_det))
    }

    /// Convert a finite `f64` constant into the target float type, returning a
    /// typed error instead of panicking when the conversion is impossible.
    ///
    /// Used by the score path for `0.5`, `2π`, and the sample/feature counts —
    /// all of which always convert cleanly for `f32`/`f64` but could in
    /// principle fail for an exotic `Float` impl (R-CODE-2: no `.unwrap`).
    fn const_f(v: f64) -> Result<F, FerroError> {
        F::from(v).ok_or_else(|| FerroError::NumericalInstability {
            message: "failed to convert a constant into the target float type".into(),
        })
    }

    /// Return the per-sample Gaussian log-likelihood under the probabilistic-PCA
    /// model.
    ///
    /// For centered rows `Xr = X − mean_` and `precision = get_precision()`,
    /// each `ll_i = −0.5 · (Xr_i · precision · Xr_iᵀ)
    ///            − 0.5 · (p · ln(2π) − logdet(precision))`, where
    /// `p = n_features` and `logdet(precision) = −Σ ln(λ)` (λ = eigenvalues of
    /// `get_covariance`). Mirrors sklearn `PCA.score_samples`
    /// (`sklearn/decomposition/_pca.py:805-830`:
    /// `log_like = -0.5 * sum(Xr * (Xr @ precision), axis=1)`;
    /// `log_like -= 0.5 * (n_features * log(2π) - fast_logdet(precision))`).
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] if `x.ncols()` differs from the number of
    ///   features seen during fitting.
    /// - [`FerroError::NumericalInstability`] propagated from
    ///   [`get_precision`](FittedPCA::get_precision) (`whiten` enabled,
    ///   eigendecomposition failure, or a non-positive covariance eigenvalue).
    pub fn score_samples(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        let n_features = self.mean_.len();
        if x.ncols() != n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedPCA::score_samples".into(),
            });
        }

        let (precision, log_det) = self.precision_and_logdet()?;

        // Constant term: 0.5 * (p·ln(2π) − logdet(precision)).
        let two_pi = Self::const_f(2.0 * std::f64::consts::PI)?;
        let half = Self::const_f(0.5)?;
        let p = Self::const_f(n_features as f64)?;
        let const_term = half * (p * two_pi.ln() - log_det);

        let mut ll = Array1::<F>::zeros(x.nrows());
        for (i, row) in x.rows().into_iter().enumerate() {
            // Xr_i = row − mean_
            let mut xr = Array1::<F>::zeros(n_features);
            for j in 0..n_features {
                xr[j] = row[j] - self.mean_[j];
            }
            // quad = Xr_i · precision · Xr_iᵀ
            let mut quad = F::zero();
            for a in 0..n_features {
                let mut pa = F::zero();
                for b in 0..n_features {
                    pa = pa + precision[[a, b]] * xr[b];
                }
                quad = quad + xr[a] * pa;
            }
            ll[i] = -half * quad - const_term;
        }

        Ok(ll)
    }

    /// Return the average Gaussian log-likelihood of all samples under the
    /// probabilistic-PCA model: `mean(score_samples(X))`.
    ///
    /// Mirrors sklearn `PCA.score` (`sklearn/decomposition/_pca.py:832-853`:
    /// `float(mean(self.score_samples(X)))`).
    ///
    /// # Errors
    ///
    /// Propagates the errors of [`score_samples`](FittedPCA::score_samples).
    pub fn score(&self, x: &Array2<F>) -> Result<F, FerroError> {
        let ll = self.score_samples(x)?;
        let n = Self::const_f(ll.len() as f64)?;
        let sum = ll.iter().copied().fold(F::zero(), |a, b| a + b);
        Ok(sum / n)
    }

    /// Reconstruct approximate original data from the reduced representation.
    ///
    /// Computes `X_approx = X_reduced @ components + mean`. When whitening is
    /// enabled, each input column j is first multiplied by
    /// `sqrt(explained_variance_[j])` to reverse the `transform` scaling,
    /// mirroring sklearn `inverse_transform`
    /// (`sklearn/decomposition/_base.py:192-196`:
    /// `scaled_components = sqrt(explained_variance_)[:, newaxis] * components_;
    /// X @ scaled_components + mean_`).
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns in
    /// `x_reduced` does not equal `n_components`.
    pub fn inverse_transform(&self, x_reduced: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_components = self.components_.nrows();
        if x_reduced.ncols() != n_components {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x_reduced.nrows(), n_components],
                actual: vec![x_reduced.nrows(), x_reduced.ncols()],
                context: "FittedPCA::inverse_transform".into(),
            });
        }

        // Whitening: multiply each column j by sqrt(explained_variance_[j])
        // BEFORE projecting back, reversing the `transform` divide. This is
        // algebraically identical to sklearn folding the scale into the
        // components (`_base.py:192-196`). When `whiten` is false the input is
        // used unchanged.
        let x_scaled = if self.whiten {
            let mut x_scaled = x_reduced.to_owned();
            for j in 0..n_components {
                let scale = self.explained_variance_[j].sqrt();
                for v in x_scaled.column_mut(j) {
                    *v = *v * scale;
                }
            }
            std::borrow::Cow::Owned(x_scaled)
        } else {
            std::borrow::Cow::Borrowed(x_reduced)
        };

        // X_approx = X_scaled @ components + mean
        let mut result = x_scaled.dot(&self.components_);
        for mut row in result.rows_mut() {
            for (v, &m) in row.iter_mut().zip(self.mean_.iter()) {
                *v = *v + m;
            }
        }
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Jacobi eigendecomposition for symmetric matrices
// ---------------------------------------------------------------------------

/// Perform eigendecomposition of a symmetric matrix using the Jacobi method.
///
/// Returns `(eigenvalues, eigenvectors)` where `eigenvectors` is column-major
/// (column `i` is the eigenvector for `eigenvalues[i]`).
///
/// The eigenvalues are NOT sorted; the caller is responsible for sorting.
fn jacobi_eigen<F: Float + Send + Sync + 'static>(
    a: &Array2<F>,
    max_iter: usize,
) -> Result<(Array1<F>, Array2<F>), FerroError> {
    let n = a.nrows();
    let mut mat = a.to_owned();
    let mut v = Array2::<F>::zeros((n, n));
    // Initialise V to identity.
    for i in 0..n {
        v[[i, i]] = F::one();
    }

    let tol = F::from(1e-12).unwrap_or_else(F::epsilon);

    for iteration in 0..max_iter {
        // Find the largest off-diagonal element.
        let mut max_off = F::zero();
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                let val = mat[[i, j]].abs();
                if val > max_off {
                    max_off = val;
                    p = i;
                    q = j;
                }
            }
        }

        if max_off < tol {
            // Converged.
            let eigenvalues = Array1::from_shape_fn(n, |i| mat[[i, i]]);
            return Ok((eigenvalues, v));
        }

        // Compute the Jacobi rotation.
        let app = mat[[p, p]];
        let aqq = mat[[q, q]];
        let apq = mat[[p, q]];

        let theta = if (app - aqq).abs() < tol {
            F::from(std::f64::consts::FRAC_PI_4).unwrap_or_else(F::one)
        } else {
            let tau = (aqq - app) / (F::from(2.0).unwrap() * apq);
            // t = sign(tau) / (|tau| + sqrt(1 + tau^2))
            let t = if tau >= F::zero() {
                F::one() / (tau.abs() + (F::one() + tau * tau).sqrt())
            } else {
                -F::one() / (tau.abs() + (F::one() + tau * tau).sqrt())
            };
            t.atan()
        };

        let c = theta.cos();
        let s = theta.sin();

        // Apply rotation to mat: mat' = G^T mat G
        // Update rows/columns p and q.
        let mut new_mat = mat.clone();

        for i in 0..n {
            if i != p && i != q {
                let mip = mat[[i, p]];
                let miq = mat[[i, q]];
                new_mat[[i, p]] = c * mip - s * miq;
                new_mat[[p, i]] = new_mat[[i, p]];
                new_mat[[i, q]] = s * mip + c * miq;
                new_mat[[q, i]] = new_mat[[i, q]];
            }
        }

        new_mat[[p, p]] = c * c * app - F::from(2.0).unwrap() * s * c * apq + s * s * aqq;
        new_mat[[q, q]] = s * s * app + F::from(2.0).unwrap() * s * c * apq + c * c * aqq;
        new_mat[[p, q]] = F::zero();
        new_mat[[q, p]] = F::zero();

        mat = new_mat;

        // Update eigenvector matrix.
        for i in 0..n {
            let vip = v[[i, p]];
            let viq = v[[i, q]];
            v[[i, p]] = c * vip - s * viq;
            v[[i, q]] = s * vip + c * viq;
        }

        let _ = iteration; // suppress unused warning
    }

    Err(FerroError::ConvergenceFailure {
        iterations: max_iter,
        message: "Jacobi eigendecomposition did not converge".into(),
    })
}

// ---------------------------------------------------------------------------
// faer-accelerated eigendecomposition for f64 and f32
// ---------------------------------------------------------------------------

/// Perform eigendecomposition of a symmetric matrix using faer's optimised
/// self-adjoint eigensolver. Returns `(eigenvalues, eigenvectors)` where
/// `eigenvectors` is column-major (column `i` is the eigenvector for
/// `eigenvalues[i]`). Eigenvalues are returned in ascending order.
fn faer_eigen_f64(a: &Array2<f64>) -> Result<(Array1<f64>, Array2<f64>), FerroError> {
    let n = a.nrows();
    let mat = faer::Mat::from_fn(n, n, |i, j| a[[i, j]]);
    let decomp = mat.self_adjoint_eigen(faer::Side::Lower).map_err(|e| {
        FerroError::NumericalInstability {
            message: format!("faer symmetric eigendecomposition failed: {e:?}"),
        }
    })?;

    let eigenvalues = Array1::from_shape_fn(n, |i| decomp.S().column_vector()[i]);
    let eigenvectors = Array2::from_shape_fn((n, n), |(i, j)| decomp.U()[(i, j)]);

    Ok((eigenvalues, eigenvectors))
}

/// Perform eigendecomposition of a symmetric f32 matrix using faer's
/// optimised self-adjoint eigensolver.
fn faer_eigen_f32(a: &Array2<f32>) -> Result<(Array1<f32>, Array2<f32>), FerroError> {
    let n = a.nrows();
    let mat = faer::Mat::from_fn(n, n, |i, j| a[[i, j]]);
    let decomp = mat.self_adjoint_eigen(faer::Side::Lower).map_err(|e| {
        FerroError::NumericalInstability {
            message: format!("faer symmetric eigendecomposition failed: {e:?}"),
        }
    })?;

    let eigenvalues = Array1::from_shape_fn(n, |i| decomp.S().column_vector()[i]);
    let eigenvectors = Array2::from_shape_fn((n, n), |(i, j)| decomp.U()[(i, j)]);

    Ok((eigenvalues, eigenvectors))
}

/// Dispatch eigendecomposition to faer for f64/f32, falling back to
/// Jacobi for other float types.
///
/// Returns `(eigenvalues, eigenvectors)` where `eigenvectors` is column-major.
/// Eigenvalues are NOT guaranteed to be sorted in any particular order.
fn eigen_dispatch<F: Float + Send + Sync + 'static>(
    a: &Array2<F>,
    max_jacobi_iter: usize,
) -> Result<(Array1<F>, Array2<F>), FerroError> {
    // SAFETY: We check TypeId at runtime and only reinterpret when the types
    // match. The transmutes are between identical types (Array<f64> -> Array<F>
    // when F == f64, etc.).
    if TypeId::of::<F>() == TypeId::of::<f64>() {
        // F is f64 — cast through raw pointers to call the f64 specialisation.
        let a_f64: &Array2<f64> = unsafe { &*(std::ptr::from_ref(a).cast::<Array2<f64>>()) };
        let (eigenvalues, eigenvectors) = faer_eigen_f64(a_f64)?;
        // Cast back from f64 to F (which is f64).
        let eigenvalues_f: Array1<F> =
            unsafe { std::mem::transmute_copy::<Array1<f64>, Array1<F>>(&eigenvalues) };
        let eigenvectors_f: Array2<F> =
            unsafe { std::mem::transmute_copy::<Array2<f64>, Array2<F>>(&eigenvectors) };
        // Prevent double-free of the originals.
        std::mem::forget(eigenvalues);
        std::mem::forget(eigenvectors);
        Ok((eigenvalues_f, eigenvectors_f))
    } else if TypeId::of::<F>() == TypeId::of::<f32>() {
        let a_f32: &Array2<f32> = unsafe { &*(std::ptr::from_ref(a).cast::<Array2<f32>>()) };
        let (eigenvalues, eigenvectors) = faer_eigen_f32(a_f32)?;
        let eigenvalues_f: Array1<F> =
            unsafe { std::mem::transmute_copy::<Array1<f32>, Array1<F>>(&eigenvalues) };
        let eigenvectors_f: Array2<F> =
            unsafe { std::mem::transmute_copy::<Array2<f32>, Array2<F>>(&eigenvectors) };
        std::mem::forget(eigenvalues);
        std::mem::forget(eigenvectors);
        Ok((eigenvalues_f, eigenvectors_f))
    } else {
        // Fallback to Jacobi for exotic float types.
        jacobi_eigen(a, max_jacobi_iter)
    }
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for PCA<F> {
    type Fitted = FittedPCA<F>;
    type Error = FerroError;

    /// Fit PCA by centring the data and eigendecomposing the covariance matrix.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `n_components` is zero or exceeds
    ///   the number of features.
    /// - [`FerroError::InsufficientSamples`] if there are fewer than 2 samples.
    /// - [`FerroError::ConvergenceFailure`] if the Jacobi eigendecomposition
    ///   does not converge.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedPCA<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if self.n_components == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: "must be at least 1".into(),
            });
        }
        if self.n_components > n_features {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: format!(
                    "n_components ({}) exceeds n_features ({})",
                    self.n_components, n_features
                ),
            });
        }
        if n_samples < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: n_samples,
                context: "PCA::fit requires at least 2 samples".into(),
            });
        }

        let n_f = F::from(n_samples).unwrap();

        // Step 1: compute mean and centre data.
        let mut mean = Array1::<F>::zeros(n_features);
        for j in 0..n_features {
            let col = x.column(j);
            let sum = col.iter().copied().fold(F::zero(), |a, b| a + b);
            mean[j] = sum / n_f;
        }

        let mut x_centered = x.to_owned();
        for mut row in x_centered.rows_mut() {
            for (v, &m) in row.iter_mut().zip(mean.iter()) {
                *v = *v - m;
            }
        }

        // Step 2: compute covariance matrix C = X_centered^T @ X_centered / (n-1)
        let n_minus_1 = F::from(n_samples - 1).unwrap();
        let xt = x_centered.t();
        let mut cov = xt.dot(&x_centered);
        cov.mapv_inplace(|v| v / n_minus_1);

        // Step 3: eigendecompose (faer fast-path for f64/f32, Jacobi fallback)
        let max_jacobi_iter = n_features * n_features * 100 + 1000;
        let (eigenvalues, eigenvectors) = eigen_dispatch(&cov, max_jacobi_iter)?;

        // Step 4: sort eigenvalues descending and select top n_components
        let mut indices: Vec<usize> = (0..n_features).collect();
        indices.sort_by(|&a, &b| {
            eigenvalues[b]
                .partial_cmp(&eigenvalues[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let total_variance = eigenvalues.iter().copied().fold(F::zero(), |a, b| a + b);

        let n_comp = self.n_components;
        let mut components = Array2::<F>::zeros((n_comp, n_features));
        let mut explained_variance = Array1::<F>::zeros(n_comp);
        let mut explained_variance_ratio = Array1::<F>::zeros(n_comp);
        let mut singular_values = Array1::<F>::zeros(n_comp);

        for (k, &idx) in indices.iter().take(n_comp).enumerate() {
            let eigval = eigenvalues[idx];
            // Clamp small negative eigenvalues to zero (numerical noise).
            let eigval_clamped = if eigval < F::zero() {
                F::zero()
            } else {
                eigval
            };
            explained_variance[k] = eigval_clamped;
            explained_variance_ratio[k] = if total_variance > F::zero() {
                eigval_clamped / total_variance
            } else {
                F::zero()
            };
            // singular_value = sqrt(eigenvalue * (n_samples - 1))
            singular_values[k] = (eigval_clamped * n_minus_1).sqrt();

            // The eigenvector is a column of `eigenvectors`; store it as a row
            // of `components_`.
            for j in 0..n_features {
                components[[k, j]] = eigenvectors[[j, idx]];
            }

            // Sign convention: mirror sklearn `svd_flip(U, Vt,
            // u_based_decision=False)` (`_pca.py:647`, `extmath.py:897-905`).
            // For each component row, find the column with the maximum absolute
            // value (numpy `argmax` → FIRST on ties: iterate from 0 and update
            // only on STRICT `>`). If that entry is negative, negate the whole
            // row so its max-abs entry is positive. This pins the otherwise
            // arbitrary faer eigenvector signs deterministically.
            let mut j_max = 0usize;
            let mut max_abs = components[[k, 0]].abs();
            for j in 1..n_features {
                let abs_j = components[[k, j]].abs();
                if abs_j > max_abs {
                    max_abs = abs_j;
                    j_max = j;
                }
            }
            if components[[k, j_max]] < F::zero() {
                for j in 0..n_features {
                    components[[k, j]] = -components[[k, j]];
                }
            }
        }

        // noise_variance_ from the DISCARDED eigenvalue tail (probabilistic PCA,
        // sklearn `_pca.py:685-688`): if `n_comp < min(n_samples, n_features)`,
        // it is the MEAN of the full (descending-sorted) explained variances
        // from index `n_comp` up to `min_dim − 1`; otherwise 0. Because `cov`
        // is already `XᵀX/(n−1)`, its eigenvalues ARE the explained variances
        // (same `1/(n−1)` scaling as `explained_variance_`), so we average the
        // sorted eigenvalues directly (negatives clipped to 0, matching the
        // `explained_variance_` clip at `_pca.py:637`).
        let min_dim = n_samples.min(n_features);
        let noise_variance = if n_comp < min_dim {
            let mut tail_sum = F::zero();
            for &idx in indices.iter().take(min_dim).skip(n_comp) {
                let eigval = eigenvalues[idx];
                let eigval_clamped = if eigval < F::zero() {
                    F::zero()
                } else {
                    eigval
                };
                tail_sum = tail_sum + eigval_clamped;
            }
            let count = FittedPCA::<F>::const_f((min_dim - n_comp) as f64)?;
            tail_sum / count
        } else {
            F::zero()
        };

        Ok(FittedPCA {
            components_: components,
            explained_variance_: explained_variance,
            explained_variance_ratio_: explained_variance_ratio,
            mean_: mean,
            singular_values_: singular_values,
            noise_variance_: noise_variance,
            whiten: self.whiten,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedPCA<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Project data onto the principal components: `(X - mean) @ components^T`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns does not
    /// match the number of features seen during fitting.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features = self.mean_.len();
        if x.ncols() != n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedPCA::transform".into(),
            });
        }

        // Centre the data.
        let mut x_centered = x.to_owned();
        for mut row in x_centered.rows_mut() {
            for (v, &m) in row.iter_mut().zip(self.mean_.iter()) {
                *v = *v - m;
            }
        }

        // Project: X_centered @ components^T
        let mut result = x_centered.dot(&self.components_.t());

        // Whitening: divide each column j by sqrt(explained_variance_[j]) so
        // the transformed output has unit component-wise variance. Mirrors
        // sklearn `_transform` (`sklearn/decomposition/_base.py:157-165`):
        // `scale = sqrt(explained_variance_); scale[scale < eps] = eps;
        // X_transformed /= scale`. The eps clip guards against components with
        // a variance arbitrarily close to zero on rank-deficient data
        // (`_base.py:158-164`). When `whiten` is false this block is skipped,
        // leaving the result byte-identical to the plain projection.
        if self.whiten {
            let min_scale = F::epsilon();
            for j in 0..result.ncols() {
                let mut scale = self.explained_variance_[j].sqrt();
                if scale < min_scale {
                    scale = min_scale;
                }
                for v in result.column_mut(j) {
                    *v = *v / scale;
                }
            }
        }

        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Pipeline integration (generic)
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> PipelineTransformer<F> for PCA<F> {
    /// Fit PCA using the pipeline interface.
    ///
    /// The `y` argument is ignored; PCA is unsupervised.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Fit::fit`].
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        _y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineTransformer<F>>, FerroError> {
        let fitted = self.fit(x, &())?;
        Ok(Box::new(fitted))
    }
}

impl<F: Float + Send + Sync + 'static> FittedPipelineTransformer<F> for FittedPCA<F> {
    /// Transform data using the pipeline interface.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Transform::transform`].
    fn transform_pipeline(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        self.transform(x)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_pca_dimensionality_reduction() {
        let pca = PCA::<f64>::new(1);
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ];
        let fitted = pca.fit(&x, &()).unwrap();
        let projected = fitted.transform(&x).unwrap();
        assert_eq!(projected.dim(), (4, 1));
    }

    #[test]
    fn test_pca_explained_variance_ratio_sums_le_1() {
        let pca = PCA::<f64>::new(2);
        let x = array![
            [2.5, 2.4],
            [0.5, 0.7],
            [2.2, 2.9],
            [1.9, 2.2],
            [3.1, 3.0],
            [2.3, 2.7],
            [2.0, 1.6],
            [1.0, 1.1],
            [1.5, 1.6],
            [1.1, 0.9],
        ];
        let fitted = pca.fit(&x, &()).unwrap();
        let ratio_sum: f64 = fitted.explained_variance_ratio().iter().sum();
        // When n_components == n_features, ratio should sum to ~1.0.
        assert!(ratio_sum <= 1.0 + 1e-10, "ratio sum = {ratio_sum}");
    }

    #[test]
    fn test_pca_explained_variance_ratio_partial() {
        let pca = PCA::<f64>::new(1);
        let x = array![
            [2.5, 2.4],
            [0.5, 0.7],
            [2.2, 2.9],
            [1.9, 2.2],
            [3.1, 3.0],
            [2.3, 2.7],
            [2.0, 1.6],
            [1.0, 1.1],
            [1.5, 1.6],
            [1.1, 0.9],
        ];
        let fitted = pca.fit(&x, &()).unwrap();
        let ratio_sum: f64 = fitted.explained_variance_ratio().iter().sum();
        // With 1 component out of 2, ratio should be strictly less than 1.
        assert!(ratio_sum <= 1.0 + 1e-10);
        assert!(ratio_sum > 0.0);
    }

    #[test]
    fn test_pca_components_orthonormal() {
        let pca = PCA::<f64>::new(2);
        let x = array![
            [2.5, 2.4],
            [0.5, 0.7],
            [2.2, 2.9],
            [1.9, 2.2],
            [3.1, 3.0],
            [2.3, 2.7],
            [2.0, 1.6],
            [1.0, 1.1],
            [1.5, 1.6],
            [1.1, 0.9],
        ];
        let fitted = pca.fit(&x, &()).unwrap();
        let c = fitted.components();

        // Check that each component is unit length.
        for i in 0..c.nrows() {
            let norm: f64 = c.row(i).iter().map(|v| v * v).sum::<f64>().sqrt();
            assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-8);
        }

        // Check mutual orthogonality.
        for i in 0..c.nrows() {
            for j in (i + 1)..c.nrows() {
                let dot: f64 = c
                    .row(i)
                    .iter()
                    .zip(c.row(j).iter())
                    .map(|(a, b)| a * b)
                    .sum();
                assert_abs_diff_eq!(dot, 0.0, epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_pca_inverse_transform_roundtrip() {
        let pca = PCA::<f64>::new(2);
        let x = array![[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0],];
        let fitted = pca.fit(&x, &()).unwrap();
        let projected = fitted.transform(&x).unwrap();
        let recovered = fitted.inverse_transform(&projected).unwrap();

        // With n_components == n_features, reconstruction should be exact.
        for (a, b) in x.iter().zip(recovered.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_pca_inverse_transform_approx() {
        // With fewer components, reconstruction is lossy but the error
        // should be bounded.
        let pca = PCA::<f64>::new(1);
        let x = array![[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0],];
        let fitted = pca.fit(&x, &()).unwrap();
        let projected = fitted.transform(&x).unwrap();
        let recovered = fitted.inverse_transform(&projected).unwrap();

        // Reconstruction should not be wildly off.
        let total_error: f64 = x
            .iter()
            .zip(recovered.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        let total_var: f64 = {
            let mean_x: f64 = x.iter().sum::<f64>() / x.len() as f64;
            x.iter().map(|&v| (v - mean_x).powi(2)).sum()
        };
        // Relative reconstruction error should be reasonable.
        assert!(
            total_error < total_var,
            "error={total_error}, var={total_var}"
        );
    }

    #[test]
    fn test_pca_n_components_equals_n_features() {
        let pca = PCA::<f64>::new(3);
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ];
        let fitted = pca.fit(&x, &()).unwrap();
        let ratio_sum: f64 = fitted.explained_variance_ratio().iter().sum();
        assert_abs_diff_eq!(ratio_sum, 1.0, epsilon = 1e-8);
    }

    #[test]
    fn test_pca_single_component() {
        let pca = PCA::<f64>::new(1);
        let x = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0],];
        let fitted = pca.fit(&x, &()).unwrap();
        assert_eq!(fitted.components().nrows(), 1);
        assert_eq!(fitted.explained_variance().len(), 1);
    }

    #[test]
    fn test_pca_shape_mismatch_transform() {
        let pca = PCA::<f64>::new(1);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let fitted = pca.fit(&x, &()).unwrap();
        let x_bad = array![[1.0, 2.0, 3.0]];
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_pca_shape_mismatch_inverse_transform() {
        let pca = PCA::<f64>::new(1);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let fitted = pca.fit(&x, &()).unwrap();
        // inverse_transform expects 1 column (n_components), not 3
        let x_bad = array![[1.0, 2.0, 3.0]];
        assert!(fitted.inverse_transform(&x_bad).is_err());
    }

    #[test]
    fn test_pca_invalid_n_components_zero() {
        let pca = PCA::<f64>::new(0);
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        assert!(pca.fit(&x, &()).is_err());
    }

    #[test]
    fn test_pca_invalid_n_components_too_large() {
        let pca = PCA::<f64>::new(5);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        assert!(pca.fit(&x, &()).is_err());
    }

    #[test]
    fn test_pca_insufficient_samples() {
        let pca = PCA::<f64>::new(1);
        let x = array![[1.0, 2.0]]; // only 1 sample
        assert!(pca.fit(&x, &()).is_err());
    }

    #[test]
    fn test_pca_explained_variance_positive() {
        let pca = PCA::<f64>::new(2);
        let x = array![[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0],];
        let fitted = pca.fit(&x, &()).unwrap();
        for &v in fitted.explained_variance() {
            assert!(v >= 0.0, "negative variance: {v}");
        }
    }

    #[test]
    fn test_pca_singular_values_positive() {
        let pca = PCA::<f64>::new(2);
        let x = array![[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0],];
        let fitted = pca.fit(&x, &()).unwrap();
        for &s in fitted.singular_values() {
            assert!(s >= 0.0, "negative singular value: {s}");
        }
    }

    #[test]
    fn test_pca_f32() {
        let pca = PCA::<f32>::new(1);
        let x: Array2<f32> = array![[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0],];
        let fitted = pca.fit(&x, &()).unwrap();
        let projected = fitted.transform(&x).unwrap();
        assert_eq!(projected.ncols(), 1);
    }

    #[test]
    fn test_pca_n_components_getter() {
        let pca = PCA::<f64>::new(3);
        assert_eq!(pca.n_components(), 3);
    }

    #[test]
    fn test_pca_pipeline_integration() {
        use ferrolearn_core::pipeline::{FittedPipelineEstimator, Pipeline, PipelineEstimator};
        use ferrolearn_core::traits::Predict;

        // Trivial estimator that sums each row.
        struct SumEstimator;

        impl PipelineEstimator<f64> for SumEstimator {
            fn fit_pipeline(
                &self,
                _x: &Array2<f64>,
                _y: &Array1<f64>,
            ) -> Result<Box<dyn FittedPipelineEstimator<f64>>, FerroError> {
                Ok(Box::new(FittedSumEstimator))
            }
        }

        struct FittedSumEstimator;

        impl FittedPipelineEstimator<f64> for FittedSumEstimator {
            fn predict_pipeline(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
                let sums: Vec<f64> = x.rows().into_iter().map(|r| r.sum()).collect();
                Ok(Array1::from_vec(sums))
            }
        }

        let pipeline = Pipeline::new()
            .transform_step("pca", Box::new(PCA::<f64>::new(1)))
            .estimator_step("sum", Box::new(SumEstimator));

        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ];
        let y = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0]);

        let fitted = pipeline.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    // ---- REQ-11: whiten (sklearn _base.py:157-165,:192-196) --------------
    //
    // Oracle = live scikit-learn 1.5.2 (R-CHAR-3), run from /tmp:
    //   X = [[1,2,3],[4,5,6],[7,8,10],[2,1,0],[5,3,2]]
    //   m = PCA(n_components=2, whiten=<...>).fit(X)
    //   m.transform(X) / m.inverse_transform(m.transform(X))
    // PCA component signs are deterministic per REQ-1's svd_flip, so the
    // element-wise comparison (including sign) is valid.

    fn whiten_fixture() -> Array2<f64> {
        array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 10.0],
            [2.0, 1.0, 0.0],
            [5.0, 3.0, 2.0],
        ]
    }

    #[test]
    fn pca_whiten_transform_matches_sklearn() -> Result<(), FerroError> {
        let x = whiten_fixture();
        let fitted = PCA::<f64>::new(2).with_whiten(true).fit(&x, &())?;

        // sklearn explained_variance_ sanity (oracle [26.42340146, 2.16534729]).
        let ev = fitted.explained_variance();
        assert_abs_diff_eq!(ev[0], 26.423_401_46, epsilon = 1e-6);
        assert_abs_diff_eq!(ev[1], 2.165_347_29, epsilon = 1e-6);

        let got = fitted.transform(&x)?;
        let expected = array![
            [-0.576_684_77, -1.310_790_08],
            [0.402_024_00, -0.444_643_84],
            [1.525_646_44, 0.083_497_24],
            [-1.039_932_95, 0.253_103_31],
            [-0.311_052_72, 1.418_833_38],
        ];
        for (a, b) in got.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-6);
        }
        Ok(())
    }

    #[test]
    fn pca_whiten_false_unchanged() -> Result<(), FerroError> {
        let x = whiten_fixture();

        // whiten=false must match the no-whiten sklearn oracle.
        let fitted = PCA::<f64>::new(2).with_whiten(false).fit(&x, &())?;
        let got = fitted.transform(&x)?;
        let expected = array![
            [-2.964_372_97, -1.928_843_21],
            [2.066_552_01, -0.654_298_71],
            [7.842_386_84, 0.122_867_18],
            [-5.345_639_88, 0.372_444_53],
            [-1.598_926_00, 2.087_830_21],
        ];
        for (a, b) in got.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-6);
        }

        // Regression guard: whiten=false MUST be byte-identical to the default
        // (no-whiten) transform — the whiten block is purely additive.
        let default_fitted = PCA::<f64>::new(2).fit(&x, &())?;
        let default_got = default_fitted.transform(&x)?;
        for (a, b) in got.iter().zip(default_got.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        Ok(())
    }

    #[test]
    fn pca_whiten_inverse_matches_sklearn() -> Result<(), FerroError> {
        let x = whiten_fixture();
        let fitted = PCA::<f64>::new(2).with_whiten(true).fit(&x, &())?;

        let transformed = fitted.transform(&x)?;
        let got = fitted.inverse_transform(&transformed)?;
        let expected = array![
            [0.965_922_29, 2.092_258_20, 2.951_174_68],
            [4.045_247_61, 4.877_501_66, 6.064_829_15],
            [6.986_571_28, 8.036_355_42, 9.980_759_81],
            [2.022_846_87, 0.938_146_92, 0.032_734_18],
            [4.979_411_95, 3.055_737_80, 1.970_502_18],
        ];
        for (a, b) in got.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-6);
        }
        Ok(())
    }

    // ---- REQ-14/15: noise_variance_ / get_covariance / get_precision /
    //      score_samples / score (probabilistic-PCA chain) -----------------
    //
    // Oracle = live scikit-learn 1.5.2 (R-CHAR-3), run from /tmp:
    //   X = [[1,2,3],[4,5,6],[7,8,10],[2,1,0],[5,3,2]]
    //   m = PCA(n_components=2).fit(X)   # whiten=False (default)
    //   m.noise_variance_ / m.get_covariance() / m.get_precision()
    //   m.score_samples(X) / m.score(X)
    // n_components=2 < min(n_samples=5, n_features=3)=3, so noise_variance_ is
    // the mean of the single discarded eigenvalue tail.

    fn prob_fixture() -> Array2<f64> {
        array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 10.0],
            [2.0, 1.0, 0.0],
            [5.0, 3.0, 2.0],
        ]
    }

    #[test]
    fn pca_noise_variance_matches_sklearn() -> Result<(), FerroError> {
        let x = prob_fixture();
        let fitted = PCA::<f64>::new(2).fit(&x, &())?;
        // sklearn m.noise_variance_ = 0.011251254758681639
        assert_abs_diff_eq!(
            fitted.noise_variance(),
            0.011_251_254_758_681_639,
            epsilon = 1e-6
        );
        Ok(())
    }

    #[test]
    fn pca_get_covariance_matches_sklearn() -> Result<(), FerroError> {
        let x = prob_fixture();
        let fitted = PCA::<f64>::new(2).fit(&x, &())?;
        let cov = fitted.get_covariance();
        // sklearn m.get_covariance()
        let expected = array![[5.7, 5.7, 6.8], [5.7, 7.7, 10.55], [6.8, 10.55, 15.2],];
        assert_eq!(cov.dim(), (3, 3));
        for (a, b) in cov.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-6);
        }
        Ok(())
    }

    #[test]
    fn pca_get_precision_matches_sklearn() -> Result<(), FerroError> {
        let x = prob_fixture();
        let fitted = PCA::<f64>::new(2).fit(&x, &())?;
        let precision = fitted.get_precision()?;
        // sklearn m.get_precision()
        let expected = array![
            [8.912_621_36, -23.145_631_07, 12.077_669_9],
            [-23.145_631_07, 62.757_281_55, -33.203_883_5],
            [12.077_669_9, -33.203_883_5, 17.708_737_86],
        ];
        assert_eq!(precision.dim(), (3, 3));
        for (a, b) in precision.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-6);
        }
        Ok(())
    }

    #[test]
    fn pca_score_samples_matches_sklearn() -> Result<(), FerroError> {
        let x = prob_fixture();
        let fitted = PCA::<f64>::new(2).fit(&x, &())?;
        let ll = fitted.score_samples(&x)?;
        // sklearn m.score_samples(X)
        let expected = array![
            -4.097_758_23,
            -3.660_865_03,
            -3.787_078_62,
            -3.350_185_42,
            -3.787_078_62,
        ];
        assert_eq!(ll.len(), 5);
        for (a, b) in ll.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-6);
        }
        Ok(())
    }

    #[test]
    fn pca_score_matches_sklearn() -> Result<(), FerroError> {
        let x = prob_fixture();
        let fitted = PCA::<f64>::new(2).fit(&x, &())?;
        // sklearn m.score(X) = -3.736593186111911
        assert_abs_diff_eq!(fitted.score(&x)?, -3.736_593_186_111_911, epsilon = 1e-6);
        Ok(())
    }
}
