//! Locally Linear Embedding (LLE).
//!
//! Non-linear dimensionality reduction that preserves local geometry by
//! reconstructing each point from its nearest neighbors and then finding a
//! low-dimensional embedding that preserves those reconstruction weights.
//!
//! # Algorithm
//!
//! 1. Find k-nearest neighbors for each data point.
//! 2. Compute reconstruction weights `W` by solving local least-squares
//!    problems: for each point, minimise `||x_i - sum_j w_ij x_j||^2`
//!    subject to `sum_j w_ij = 1`.
//! 3. Construct `M = (I - W)^T (I - W)` and find the bottom `n_components`
//!    eigenvectors of `M`, excluding the trivial constant eigenvector.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_decomp::LLE;
//! use ferrolearn_core::traits::Fit;
//! use ndarray::array;
//!
//! let lle = LLE::new(2);
//! let x = array![
//!     [0.0, 0.0],
//!     [1.0, 0.0],
//!     [2.0, 0.0],
//!     [0.0, 1.0],
//!     [1.0, 1.0],
//!     [2.0, 1.0],
//! ];
//! let fitted = lle.fit(&x, &()).unwrap();
//! assert_eq!(fitted.embedding().ncols(), 2);
//! ```
//!
//! ## REQ status
//!
//! Translation target: scikit-learn 1.5.2 `class LocallyLinearEmbedding`
//! (`sklearn/manifold/_locally_linear.py`, `method='standard'`). Tracking:
//! #1459. Each REQ is BINARY — SHIPPED (impl + non-test consumer + tests + green
//! verification) or NOT-STARTED (with a concrete open blocker).
//!
//! | REQ | Scope | Status | Evidence / Blocker |
//! |-----|-------|--------|--------------------|
//! | REQ-1 | Standard-LLE embedding VALUE parity (sign-robust: bottom `M` eigenvectors) | SHIPPED | `fit` `M=(I-W)ᵀ(I-W)` bottom eigenvectors match sklearn `null_space` (`_locally_linear.py:192-196`) up to per-component sign; element-wise (sign-aligned) matches live sklearn across n_neighbors/reg/n_components/larger/higher-D fixtures (tol 1e-6) in `tests/divergence_lle.rs` (was #1460, fixed). Consumer: re-export `lib.rs:94` |
//! | REQ-2 | Reconstruction weights `W` (local least-squares + reg + normalize) | SHIPPED | `compute_weights` reg `R = reg·trace if trace>0 else reg` matches `barycenter_weights` (`_locally_linear.py:72-79`, NO /k — fixed #1460); `w/Σw` normalize; verified via the embedding parity |
//! | REQ-3 | `M = (I-W)ᵀ(I-W)` + bottom-eigenvector extraction (skip trivial) | SHIPPED | `fit`; matches `null_space(M, n_components, k_skip=1)` `_locally_linear.py:295-301` selection |
//! | REQ-4 | Structural (embedding shape, deterministic, columns centered) | SHIPPED (scoped) | shape + determinism + column-centering guards |
//! | REQ-5 | Error/parameter contracts (n_components 0/≥n, n_neighbors 0/≥n, negative reg, NON-FINITE rejection) | SHIPPED (scoped) | `fit` guards; divergence error tests. NON-FINITE: `fit` calls `reject_non_finite` (`lle.rs` symbol `reject_non_finite`) BEFORE the kNN/reconstruction-weight/eigen math, returning the CLEAN finiteness `InvalidParameter{name:"X", reason:"Input X contains NaN or infinity."}` = sklearn `_validate_data(force_all_finite=True)` (`_locally_linear.py:793`,`utils/validation.py:147-154`). `tests/divergence_nonfinite_spillover.rs::divergence_lle_fit_nan`/`_fit_inf` match the live sklearn 1.5.2 oracle (#2290). (`transform` out-of-sample is REQ-8 NOT-STARTED, so no transform gate yet.) |
//! | REQ-6 | `method` ∈ {hessian, modified, ltsa} | NOT-STARTED | standard only; sklearn `_locally_linear.py:201-460` — blocker #1461 |
//! | REQ-7 | `eigen_solver='arpack'` + `random_state` + per-component sign convention (CARVE-OUT) | NOT-STARTED | dense faer; sklearn `_locally_linear.py:173-188` — blocker #1462 |
//! | REQ-8 | `transform` out-of-sample (barycenter weights on new points) | NOT-STARTED | sklearn `_locally_linear.py:851` — blocker #1463 |
//! | REQ-9 | `reconstruction_error_`/`nbrs_`/`embedding_` attrs + `neighbors_algorithm` + `tol`/`max_iter` | NOT-STARTED | sklearn `_locally_linear.py:785` — blocker #1464 |
//! | REQ-10 | PyO3 binding | NOT-STARTED | no `ferrolearn-python` registration — blocker #1465 |
//! | REQ-11 | ferray substrate | NOT-STARTED | dense `Array2` only — blocker #1466 |

use crate::mds::eigh_faer;
use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::Fit;
use ndarray::Array2;

/// Reject non-finite input the way sklearn's `_validate_data` does.
///
/// sklearn runs `check_array` with the default `force_all_finite=True` at the
/// top of `LocallyLinearEmbedding.fit`
/// (`sklearn/manifold/_locally_linear.py:793`; out-of-sample `transform` at
/// `:872`), raising `ValueError("Input X contains NaN.")` /
/// `"... contains infinity ..."` (`sklearn/utils/validation.py:147-154`) BEFORE
/// the kNN / reconstruction-weight / eigen math. NaN AND infinity are both
/// rejected. The message names "NaN" and "infinity" to mirror sklearn's
/// `ValueError`. Never panics (R-CODE-2).
fn reject_non_finite(x: &Array2<f64>) -> Result<(), FerroError> {
    if x.iter().any(|v| !v.is_finite()) {
        return Err(FerroError::InvalidParameter {
            name: "X".into(),
            reason: "Input X contains NaN or infinity.".into(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// LLE (unfitted)
// ---------------------------------------------------------------------------

/// Locally Linear Embedding configuration.
///
/// Holds hyperparameters for the LLE algorithm. Call [`Fit::fit`] to compute
/// the embedding and obtain a [`FittedLLE`].
#[derive(Debug, Clone)]
pub struct LLE {
    /// Number of embedding dimensions.
    n_components: usize,
    /// Number of nearest neighbors.
    n_neighbors: usize,
    /// Regularization parameter added to the local covariance matrix.
    reg: f64,
}

impl LLE {
    /// Create a new `LLE` with `n_components` embedding dimensions.
    ///
    /// Defaults: `n_neighbors = 5`, `reg = 1e-3`.
    #[must_use]
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            n_neighbors: 5,
            reg: 1e-3,
        }
    }

    /// Set the number of nearest neighbors.
    #[must_use]
    pub fn with_n_neighbors(mut self, k: usize) -> Self {
        self.n_neighbors = k;
        self
    }

    /// Set the regularization parameter.
    #[must_use]
    pub fn with_reg(mut self, reg: f64) -> Self {
        self.reg = reg;
        self
    }

    /// Return the configured number of components.
    #[must_use]
    pub fn n_components(&self) -> usize {
        self.n_components
    }

    /// Return the configured number of neighbors.
    #[must_use]
    pub fn n_neighbors(&self) -> usize {
        self.n_neighbors
    }

    /// Return the configured regularization parameter.
    #[must_use]
    pub fn reg(&self) -> f64 {
        self.reg
    }
}

// ---------------------------------------------------------------------------
// FittedLLE
// ---------------------------------------------------------------------------

/// A fitted LLE model holding the learned embedding.
///
/// Created by calling [`Fit::fit`] on a [`LLE`].
#[derive(Debug, Clone)]
pub struct FittedLLE {
    /// The embedding, shape `(n_samples, n_components)`.
    embedding_: Array2<f64>,
}

impl FittedLLE {
    /// The embedding coordinates, shape `(n_samples, n_components)`.
    #[must_use]
    pub fn embedding(&self) -> &Array2<f64> {
        &self.embedding_
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Find the k-nearest neighbors for each point.
/// Returns `neighbors[i]` = sorted Vec of neighbor indices.
fn find_neighbors(x: &Array2<f64>, k: usize) -> Vec<Vec<usize>> {
    let n = x.nrows();
    let d = x.ncols();
    let mut result = Vec::with_capacity(n);

    for i in 0..n {
        let mut dists: Vec<(f64, usize)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                let mut sq = 0.0;
                for f in 0..d {
                    let diff = x[[i, f]] - x[[j, f]];
                    sq += diff * diff;
                }
                (sq, j)
            })
            .collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        result.push(dists.iter().take(k).map(|&(_, j)| j).collect());
    }
    result
}

/// Solve for reconstruction weights using local covariance + regularization.
///
/// For each point i, solve for w such that:
///   - w minimises ||x_i - sum_j w_j * x_{neighbors_j}||^2
///   - sum_j w_j = 1
///
/// Returns a sparse weight matrix stored as dense `(n, n)`.
fn compute_weights(
    x: &Array2<f64>,
    neighbors: &[Vec<usize>],
    reg: f64,
) -> Result<Array2<f64>, FerroError> {
    let n = x.nrows();
    let d = x.ncols();
    let mut w = Array2::<f64>::zeros((n, n));

    for i in 0..n {
        let k = neighbors[i].len();

        // Z[j][f] = x[i][f] - x[neighbors[j]][f]
        let mut z = Array2::<f64>::zeros((k, d));
        for (j_idx, &j) in neighbors[i].iter().enumerate() {
            for f in 0..d {
                z[[j_idx, f]] = x[[i, f]] - x[[j, f]];
            }
        }

        // Local covariance: C = Z * Z^T
        let mut c = z.dot(&z.t());

        // Regularization (sklearn barycenter_weights, _locally_linear.py:72-77):
        // trace = np.trace(G); R = reg * trace if trace > 0 else reg.
        let trace: f64 = (0..k).map(|j| c[[j, j]]).sum();
        let reg_val = if trace > 0.0 { reg * trace } else { reg };
        for j in 0..k {
            c[[j, j]] += reg_val;
        }

        // Solve C * w_local = ones(k)
        // Use a simple Gaussian elimination (the matrix is small, k x k).
        let mut augmented = Array2::<f64>::zeros((k, k + 1));
        for r in 0..k {
            for col in 0..k {
                augmented[[r, col]] = c[[r, col]];
            }
            augmented[[r, k]] = 1.0;
        }

        // Forward elimination with partial pivoting.
        for col in 0..k {
            // Find pivot.
            let mut max_val = augmented[[col, col]].abs();
            let mut max_row = col;
            for r in (col + 1)..k {
                let val = augmented[[r, col]].abs();
                if val > max_val {
                    max_val = val;
                    max_row = r;
                }
            }
            if max_val < 1e-15 {
                return Err(FerroError::NumericalInstability {
                    message: format!(
                        "Singular local covariance matrix at point {i}. \
                         Try increasing reg or n_neighbors."
                    ),
                });
            }
            if max_row != col {
                for c_idx in 0..=k {
                    let tmp = augmented[[col, c_idx]];
                    augmented[[col, c_idx]] = augmented[[max_row, c_idx]];
                    augmented[[max_row, c_idx]] = tmp;
                }
            }
            let pivot = augmented[[col, col]];
            for c_idx in col..=k {
                augmented[[col, c_idx]] /= pivot;
            }
            for r in 0..k {
                if r != col {
                    let factor = augmented[[r, col]];
                    for c_idx in col..=k {
                        augmented[[r, c_idx]] -= factor * augmented[[col, c_idx]];
                    }
                }
            }
        }

        // Extract solution.
        let mut w_local = vec![0.0; k];
        for j in 0..k {
            w_local[j] = augmented[[j, k]];
        }

        // Normalise so that sum = 1.
        let sum: f64 = w_local.iter().sum();
        if sum.abs() > 1e-15 {
            for val in &mut w_local {
                *val /= sum;
            }
        }

        // Store in the weight matrix.
        for (j_idx, &j) in neighbors[i].iter().enumerate() {
            w[[i, j]] = w_local[j_idx];
        }
    }

    Ok(w)
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl Fit<Array2<f64>, ()> for LLE {
    type Fitted = FittedLLE;
    type Error = FerroError;

    /// Fit LLE by computing reconstruction weights and finding the
    /// bottom eigenvectors of `(I - W)^T (I - W)`.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `n_components` is zero,
    ///   `n_neighbors` is zero, or parameters are out of range.
    /// - [`FerroError::InsufficientSamples`] if there are fewer than
    ///   `n_neighbors + 1` samples.
    /// - [`FerroError::NumericalInstability`] if a local covariance matrix
    ///   is singular.
    fn fit(&self, x: &Array2<f64>, _y: &()) -> Result<FittedLLE, FerroError> {
        let n = x.nrows();
        let n_features = x.ncols();

        if self.n_components == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: "must be at least 1".into(),
            });
        }
        if self.n_neighbors == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_neighbors".into(),
                reason: "must be at least 1".into(),
            });
        }
        if n < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: n,
                context: "LLE::fit requires at least 2 samples".into(),
            });
        }
        if self.n_neighbors >= n {
            return Err(FerroError::InvalidParameter {
                name: "n_neighbors".into(),
                reason: format!(
                    "n_neighbors ({}) must be less than n_samples ({})",
                    self.n_neighbors, n
                ),
            });
        }
        // sklearn rejects an output dimension larger than the input dimension
        // (`if n_components > d_in: raise ValueError(...)`,
        // `_locally_linear.py:222-225`, where `d_in = X.shape[1]`). The
        // boundary `n_components == n_features` is accepted.
        if self.n_components > n_features {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: "output dimension must be less than or equal to input dimension".into(),
            });
        }
        // Need n_components + 1 eigenvectors (skipping the trivial one).
        if self.n_components >= n {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: format!(
                    "n_components ({}) must be less than n_samples ({})",
                    self.n_components, n
                ),
            });
        }
        if self.reg < 0.0 {
            return Err(FerroError::InvalidParameter {
                name: "reg".into(),
                reason: "must be non-negative".into(),
            });
        }

        // Reject NaN/Inf BEFORE the kNN / reconstruction-weight / eigen math
        // (sklearn's `_validate_data(force_all_finite=True)` at
        // `_locally_linear.py:793`, `utils/validation.py:147-154`).
        reject_non_finite(x)?;

        // Step 1: Find neighbors.
        let neighbors = find_neighbors(x, self.n_neighbors);

        // Step 2: Compute reconstruction weights.
        let w = compute_weights(x, &neighbors, self.reg)?;

        // Step 3: Construct M = (I - W)^T (I - W).
        // I - W
        let mut iw = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            iw[[i, i]] = 1.0;
            for j in 0..n {
                iw[[i, j]] -= w[[i, j]];
            }
        }
        // M = (I-W)^T (I-W)
        let m = iw.t().dot(&iw);

        // Step 4: Eigendecompose M.
        let (eigenvalues, eigenvectors) = eigh_faer(&m)?;

        // Sort eigenvalues ascending.
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            eigenvalues[a]
                .partial_cmp(&eigenvalues[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Skip the first (smallest, ~0) eigenvector, take next n_components.
        let n_comp = self.n_components;
        let mut embedding = Array2::<f64>::zeros((n, n_comp));
        for (k, &idx) in indices.iter().skip(1).take(n_comp).enumerate() {
            for i in 0..n {
                embedding[[i, k]] = eigenvectors[[i, idx]];
            }
        }

        Ok(FittedLLE {
            embedding_: embedding,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Helper: simple 2D grid.
    fn grid_data() -> Array2<f64> {
        array![
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [0.0, 2.0],
            [1.0, 2.0],
            [2.0, 2.0],
        ]
    }

    /// Helper: line data.
    fn line_data() -> Array2<f64> {
        array![
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
            [5.0, 0.0],
        ]
    }

    #[test]
    fn test_lle_basic_shape() {
        let lle = LLE::new(2).with_n_neighbors(3);
        let x = grid_data();
        let fitted = lle.fit(&x, &()).unwrap();
        assert_eq!(fitted.embedding().dim(), (9, 2));
    }

    #[test]
    fn test_lle_1d() {
        let lle = LLE::new(1).with_n_neighbors(2);
        let x = line_data();
        let fitted = lle.fit(&x, &()).unwrap();
        assert_eq!(fitted.embedding().ncols(), 1);
    }

    #[test]
    fn test_lle_preserves_local_structure() {
        // Points on a line embedded in 1D: the embedding should roughly
        // preserve neighbor relationships.
        let lle = LLE::new(1).with_n_neighbors(2);
        let x = line_data();
        let fitted = lle.fit(&x, &()).unwrap();
        let emb = fitted.embedding();
        // Check that the embedding is monotonic (preserves ordering).
        let vals: Vec<f64> = (0..6).map(|i| emb[[i, 0]]).collect();
        let ascending = vals.windows(2).all(|w| w[0] <= w[1] + 1e-6);
        let descending = vals.windows(2).all(|w| w[0] >= w[1] - 1e-6);
        assert!(
            ascending || descending,
            "embedding should be monotonic: {vals:?}"
        );
    }

    #[test]
    fn test_lle_invalid_n_components_zero() {
        let lle = LLE::new(0);
        let x = grid_data();
        assert!(lle.fit(&x, &()).is_err());
    }

    #[test]
    fn test_lle_invalid_n_neighbors_zero() {
        let lle = LLE::new(2).with_n_neighbors(0);
        let x = grid_data();
        assert!(lle.fit(&x, &()).is_err());
    }

    #[test]
    fn test_lle_n_neighbors_too_large() {
        let lle = LLE::new(2).with_n_neighbors(100);
        let x = grid_data(); // 9 samples
        assert!(lle.fit(&x, &()).is_err());
    }

    #[test]
    fn test_lle_insufficient_samples() {
        let lle = LLE::new(1).with_n_neighbors(1);
        let x = array![[1.0, 2.0]]; // 1 sample
        assert!(lle.fit(&x, &()).is_err());
    }

    #[test]
    fn test_lle_getters() {
        let lle = LLE::new(3).with_n_neighbors(7).with_reg(0.01);
        assert_eq!(lle.n_components(), 3);
        assert_eq!(lle.n_neighbors(), 7);
        assert!((lle.reg() - 0.01).abs() < 1e-15);
    }

    #[test]
    fn test_lle_default_params() {
        let lle = LLE::new(2);
        assert_eq!(lle.n_neighbors(), 5);
        assert!((lle.reg() - 1e-3).abs() < 1e-15);
    }

    #[test]
    fn test_lle_n_components_too_large() {
        let lle = LLE::new(50);
        let x = grid_data(); // 9 samples
        assert!(lle.fit(&x, &()).is_err());
    }

    #[test]
    fn test_lle_negative_reg() {
        let lle = LLE::new(2).with_reg(-1.0);
        let x = grid_data();
        assert!(lle.fit(&x, &()).is_err());
    }

    #[test]
    fn test_lle_larger_dataset() {
        let n = 20;
        let d = 3;
        let mut data = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            for j in 0..d {
                data[[i, j]] = (i * d + j) as f64 / (n * d) as f64;
            }
        }
        let lle = LLE::new(2).with_n_neighbors(5);
        let fitted = lle.fit(&data, &()).unwrap();
        assert_eq!(fitted.embedding().dim(), (20, 2));
    }

    #[test]
    fn test_lle_different_n_neighbors() {
        // Different n_neighbors should produce different embeddings.
        let x = grid_data();
        let lle3 = LLE::new(2).with_n_neighbors(3);
        let lle6 = LLE::new(2).with_n_neighbors(6);
        let fitted3 = lle3.fit(&x, &()).unwrap();
        let fitted6 = lle6.fit(&x, &()).unwrap();
        let emb3 = fitted3.embedding();
        let emb6 = fitted6.embedding();
        let mut diff_sum = 0.0;
        for (a, b) in emb3.iter().zip(emb6.iter()) {
            diff_sum += (a - b).abs();
        }
        assert!(
            diff_sum > 1e-10,
            "different n_neighbors should produce different embeddings (got diff_sum={diff_sum})"
        );
    }
}
