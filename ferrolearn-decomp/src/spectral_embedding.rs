//! Spectral Embedding (Laplacian Eigenmaps).
//!
//! Non-linear dimensionality reduction via the graph Laplacian. Points that
//! are close in the original space (according to an affinity measure) are
//! mapped to nearby points in the embedding.
//!
//! # Algorithm
//!
//! 1. Build an affinity matrix (RBF kernel or k-nearest-neighbors).
//! 2. Compute the normalised graph Laplacian:
//!    `L_sym = I - D^{-1/2} W D^{-1/2}`.
//! 3. Find the bottom-k eigenvectors of `L_sym`, excluding the trivial
//!    constant eigenvector (eigenvalue 0).
//!
//! # Examples
//!
//! ```
//! use ferrolearn_decomp::{SpectralEmbedding, Affinity};
//! use ferrolearn_core::traits::Fit;
//! use ndarray::array;
//!
//! let se = SpectralEmbedding::new(2);
//! let x = array![
//!     [0.0, 0.0],
//!     [1.0, 0.0],
//!     [0.0, 1.0],
//!     [1.0, 1.0],
//!     [5.0, 5.0],
//!     [6.0, 5.0],
//!     [5.0, 6.0],
//!     [6.0, 6.0],
//! ];
//! let fitted = se.fit(&x, &()).unwrap();
//! assert_eq!(fitted.embedding().ncols(), 2);
//! ```
//!
//! ## REQ status
//!
//! Translation target: scikit-learn 1.5.2 `class SpectralEmbedding` +
//! `spectral_embedding` (`sklearn/manifold/_spectral_embedding.py`). Tracking:
//! #1443. Each REQ is BINARY — SHIPPED (impl + non-test consumer + tests + green
//! verification) or NOT-STARTED (with a concrete open blocker).
//!
//! | REQ | Scope | Status | Evidence / Blocker |
//! |-----|-------|--------|--------------------|
//! | REQ-1 | RBF embedding VALUE parity (distinct eigenvalues) — eigenvectors of `L_sym`, `1/dd` rescale, deterministic sign-flip | SHIPPED | [`SpectralEmbedding::fit`] rescales by `1/dd` (`degree`=off-diagonal row-sum, matching scipy `csgraph_laplacian` which ignores self-loops, `_spectral_embedding.py:443`) + per-column sign-flip (`:465`); element-wise matches live sklearn (gamma=0.3 asymmetric fixture, tol 1e-6) in `tests/divergence_spectral_embedding.rs` (was #1444, fixed). Consumer: re-export `lib.rs:100` |
//! | REQ-2 | Structural embedding (shape `(n_samples, n_components)`, well-separated clusters separate, deterministic) | SHIPPED (scoped) | in-module + divergence structural guards |
//! | REQ-3 | Normalized symmetric Laplacian `L_sym = I - D^{-1/2} W D^{-1/2}` (diagonal excluded, matching scipy) | SHIPPED | `normalised_laplacian`; matches `csgraph_laplacian(normed=True)` |
//! | REQ-4 | RBF affinity off-diagonal `exp(-gamma·‖·‖²)` | SHIPPED (scoped) | `build_affinity_matrix` matches `rbf_kernel` off-diagonal (diagonal correctly 0 for the embedding Laplacian) |
//! | REQ-5 | Error/parameter contracts (n_components 0/≥n, <2 samples, kNN n_neighbors 0/≥n, gamma≤0, NON-FINITE rejection) | SHIPPED (scoped) | `fit` guards; divergence error tests. NON-FINITE: `fit` calls `reject_non_finite` (`spectral_embedding.rs` symbol `reject_non_finite`) BEFORE building the affinity matrix / Laplacian eigendecomposition, returning the CLEAN finiteness `InvalidParameter{name:"X", reason:"Input X contains NaN or infinity."}` = sklearn `_validate_data(force_all_finite=True)` (`_spectral_embedding.py:741`,`utils/validation.py:147-154`). `tests/divergence_nonfinite_spillover.rs::divergence_spectral_embedding_fit_nan`/`_fit_inf` match the live sklearn 1.5.2 oracle (#2290) |
//! | REQ-6 | kNN affinity GRAPH parity (`kneighbors_graph(include_self=True)` symmetrized `0.5(A+Aᵀ)`) | NOT-STARTED | sklearn `_spectral_embedding.py:689-710` — blocker #1445 |
//! | REQ-7 | `eigen_solver`/`random_state` + degenerate-eigenvalue subspace basis (CARVE-OUT) + symmetric-fixture sign-flip ULP tie | NOT-STARTED | sklearn `_spectral_embedding.py:347-465` — blocker #1446 |
//! | REQ-8 | Default `affinity='nearest_neighbors'` + `gamma=None→1/n_features` + precomputed/callable affinity | NOT-STARTED | sklearn `_spectral_embedding.py:660-715` — blocker #1447 |
//! | REQ-9 | `affinity_matrix_`/`n_neighbors_` fitted attrs + transform out-of-sample | NOT-STARTED | sklearn `_spectral_embedding.py:472` — blocker #1448 |
//! | REQ-10 | PyO3 binding | NOT-STARTED | no `ferrolearn-python` registration — blocker #1449 |
//! | REQ-11 | ferray substrate | NOT-STARTED | dense `Array2` only — blocker #1450 |

use crate::mds::eigh_faer;
use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::Fit;
use ndarray::Array2;

/// Reject non-finite input the way sklearn's `_validate_data` does.
///
/// sklearn runs `check_array` with the default `force_all_finite=True` at the
/// top of `SpectralEmbedding.fit`
/// (`sklearn/manifold/_spectral_embedding.py:741`), raising
/// `ValueError("Input X contains NaN.")` / `"... contains infinity ..."`
/// (`sklearn/utils/validation.py:147-154`) BEFORE building the affinity matrix
/// and the Laplacian eigendecomposition. NaN AND infinity are both rejected.
/// The message names "NaN" and "infinity" to mirror sklearn's `ValueError`.
/// Never panics (R-CODE-2).
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
// Affinity type
// ---------------------------------------------------------------------------

/// The affinity function for building the weight matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Affinity {
    /// RBF (Gaussian) kernel: `W_ij = exp(-gamma * ||x_i - x_j||^2)`.
    RBF {
        /// Kernel bandwidth parameter.
        gamma: f64,
    },
    /// k-nearest-neighbors: `W_ij = 1` if `j` is among the k nearest
    /// neighbors of `i` (or vice versa), `0` otherwise.
    NearestNeighbors {
        /// Number of neighbors.
        n_neighbors: usize,
    },
}

// ---------------------------------------------------------------------------
// SpectralEmbedding (unfitted)
// ---------------------------------------------------------------------------

/// Spectral Embedding (Laplacian Eigenmaps) configuration.
///
/// Holds hyperparameters for the spectral embedding algorithm. Call
/// [`Fit::fit`] to compute the Laplacian eigenmap and obtain a
/// [`FittedSpectralEmbedding`].
#[derive(Debug, Clone)]
pub struct SpectralEmbedding {
    /// Number of embedding dimensions.
    n_components: usize,
    /// Affinity function.
    affinity: Affinity,
}

impl SpectralEmbedding {
    /// Create a new `SpectralEmbedding` with `n_components` dimensions.
    ///
    /// The default affinity is `RBF { gamma: 1.0 }`.
    #[must_use]
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            affinity: Affinity::RBF { gamma: 1.0 },
        }
    }

    /// Set the affinity function.
    #[must_use]
    pub fn with_affinity(mut self, affinity: Affinity) -> Self {
        self.affinity = affinity;
        self
    }

    /// Return the configured number of components.
    #[must_use]
    pub fn n_components(&self) -> usize {
        self.n_components
    }

    /// Return the configured affinity.
    #[must_use]
    pub fn affinity(&self) -> Affinity {
        self.affinity
    }
}

// ---------------------------------------------------------------------------
// FittedSpectralEmbedding
// ---------------------------------------------------------------------------

/// A fitted Spectral Embedding model holding the learned embedding.
///
/// Created by calling [`Fit::fit`] on a [`SpectralEmbedding`].
#[derive(Debug, Clone)]
pub struct FittedSpectralEmbedding {
    /// The embedding, shape `(n_samples, n_components)`.
    embedding_: Array2<f64>,
}

impl FittedSpectralEmbedding {
    /// The embedding coordinates, shape `(n_samples, n_components)`.
    #[must_use]
    pub fn embedding(&self) -> &Array2<f64> {
        &self.embedding_
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build the affinity matrix.
fn build_affinity_matrix(x: &Array2<f64>, affinity: &Affinity) -> Array2<f64> {
    let n = x.nrows();
    match affinity {
        Affinity::RBF { gamma } => {
            let mut w = Array2::<f64>::zeros((n, n));
            for i in 0..n {
                for j in (i + 1)..n {
                    let mut sq = 0.0;
                    for k in 0..x.ncols() {
                        let diff = x[[i, k]] - x[[j, k]];
                        sq += diff * diff;
                    }
                    let val = (-gamma * sq).exp();
                    w[[i, j]] = val;
                    w[[j, i]] = val;
                }
                // Diagonal is 0 (no self-loops). scipy's
                // `csgraph_laplacian(W, normed=True)` treats W as a graph
                // adjacency and ignores the diagonal, computing
                // `degree[i] = Σ_{j≠i} W[i,j]`. Setting `W_ii=0` makes the
                // degree/Laplacian/`dd` match scipy (and hence sklearn) exactly.
            }
            w
        }
        Affinity::NearestNeighbors { n_neighbors } => {
            // Mirror sklearn's `kneighbors_graph(X, n_neighbors_,
            // include_self=True, mode='connectivity')` followed by the
            // `0.5 * (A + A.T)` symmetrization
            // (`sklearn/manifold/_spectral_embedding.py:703-709`).
            //
            // `include_self=True` marks each sample as the FIRST of its own
            // `n_neighbors` neighbors (`sklearn/neighbors/_graph.py:34-43`,
            // `_query_include_self`), so each row of the connectivity matrix
            // `A` has EXACTLY `k` ones: the point itself plus the `k - 1`
            // nearest OTHER points. Clamp `k` to `n` so we never overrun the
            // sample count (R-CODE-2: never panic — `fit` already rejects
            // `n_neighbors >= n`, this is defence in depth).
            let k = (*n_neighbors).min(n);
            // Compute all pairwise squared distances.
            let mut sq_dist = Array2::<f64>::zeros((n, n));
            for i in 0..n {
                for j in (i + 1)..n {
                    let mut sq = 0.0;
                    for f in 0..x.ncols() {
                        let diff = x[[i, f]] - x[[j, f]];
                        sq += diff * diff;
                    }
                    sq_dist[[i, j]] = sq;
                    sq_dist[[j, i]] = sq;
                }
            }
            // Build the (asymmetric) connectivity matrix `A`: for each point
            // `i`, set `A[i][i] = 1` (self) and `A[i][j] = 1` for the `k - 1`
            // nearest OTHER points `j`. Ties are broken by ascending index
            // (a STABLE sort over index-ordered candidates), matching numpy's
            // `argpartition`/`argsort` stable tie-break.
            let mut a = Array2::<f64>::zeros((n, n));
            for i in 0..n {
                a[[i, i]] = 1.0; // self is the first neighbor (include_self=True)
                let mut others: Vec<(f64, usize)> = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| (sq_dist[[i, j]], j))
                    .collect();
                // Stable sort by distance; equal distances keep ascending-index
                // order (the Vec is built in ascending `j` order).
                others.sort_by(|p, q| p.0.partial_cmp(&q.0).unwrap_or(std::cmp::Ordering::Equal));
                for &(_, j) in others.iter().take(k.saturating_sub(1)) {
                    a[[i, j]] = 1.0;
                }
            }
            // Symmetrize by AVERAGING: `0.5 * (A + A.T)`. A one-directional
            // edge → 0.5, a bidirectional edge → 1.0, the self-diagonal → 1.0.
            let mut w = Array2::<f64>::zeros((n, n));
            for i in 0..n {
                for j in 0..n {
                    w[[i, j]] = 0.5 * (a[[i, j]] + a[[j, i]]);
                }
            }
            // Zero the self-loop diagonal. scipy's `csgraph_laplacian(normed=True)`
            // (which sklearn applies to the symmetrized affinity) IGNORES the
            // matrix diagonal — the degree is the OFF-diagonal row-sum — so
            // dropping the `1.0` self-loop here makes the downstream degree /
            // `dd` / normalized Laplacian match scipy (and hence sklearn)
            // exactly, exactly as the RBF arm relies on (`W_ii = 0`).
            for i in 0..n {
                w[[i, i]] = 0.0;
            }
            w
        }
    }
}

/// Compute the normalised graph Laplacian `L_sym = I - D^{-1/2} W D^{-1/2}`.
fn normalised_laplacian(w: &Array2<f64>) -> Array2<f64> {
    let n = w.nrows();
    // Degree vector.
    let mut d_inv_sqrt = vec![0.0; n];
    for i in 0..n {
        let deg: f64 = (0..n).map(|j| w[[i, j]]).sum();
        d_inv_sqrt[i] = if deg > 1e-15 { 1.0 / deg.sqrt() } else { 0.0 };
    }

    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            if i == j {
                l[[i, j]] = 1.0 - d_inv_sqrt[i] * w[[i, j]] * d_inv_sqrt[j];
            } else {
                l[[i, j]] = -d_inv_sqrt[i] * w[[i, j]] * d_inv_sqrt[j];
            }
        }
    }
    l
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl Fit<Array2<f64>, ()> for SpectralEmbedding {
    type Fitted = FittedSpectralEmbedding;
    type Error = FerroError;

    /// Fit spectral embedding by building the affinity matrix, computing the
    /// normalized Laplacian, and extracting the bottom eigenvectors.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `n_components` is zero or too large.
    /// - [`FerroError::InsufficientSamples`] if there are fewer than 2 samples.
    fn fit(&self, x: &Array2<f64>, _y: &()) -> Result<FittedSpectralEmbedding, FerroError> {
        let n = x.nrows();

        if self.n_components == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: "must be at least 1".into(),
            });
        }
        if n < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: n,
                context: "SpectralEmbedding::fit requires at least 2 samples".into(),
            });
        }
        // Reject NaN/Inf BEFORE building the affinity matrix and the Laplacian
        // eigendecomposition (sklearn's `_validate_data(force_all_finite=True)`
        // at `_spectral_embedding.py:741`, `utils/validation.py:147-154`).
        reject_non_finite(x)?;
        // We need n_components + 1 eigenvectors (to skip the trivial one).
        if self.n_components >= n {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: format!(
                    "n_components ({}) must be less than n_samples ({})",
                    self.n_components, n
                ),
            });
        }

        if let Affinity::NearestNeighbors { n_neighbors } = self.affinity {
            if n_neighbors == 0 {
                return Err(FerroError::InvalidParameter {
                    name: "n_neighbors".into(),
                    reason: "must be at least 1".into(),
                });
            }
            if n_neighbors >= n {
                return Err(FerroError::InvalidParameter {
                    name: "n_neighbors".into(),
                    reason: format!(
                        "n_neighbors ({n_neighbors}) must be less than n_samples ({n})"
                    ),
                });
            }
        }

        if let Affinity::RBF { gamma } = self.affinity
            && gamma <= 0.0
        {
            return Err(FerroError::InvalidParameter {
                name: "gamma".into(),
                reason: "must be positive".into(),
            });
        }

        // Step 1: Build affinity matrix.
        let w = build_affinity_matrix(x, &self.affinity);

        // Step 2: Normalised Laplacian.
        let l = normalised_laplacian(&w);

        // Step 3: Eigendecompose (faer returns sorted ascending eigenvalues).
        let (eigenvalues, eigenvectors) = eigh_faer(&l)?;

        // Sort eigenvalues ascending (they should already be, but be safe).
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            eigenvalues[a]
                .partial_cmp(&eigenvalues[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // sklearn rescales the eigenvectors by `dd = sqrt(degree)`
        // (`_spectral_embedding.py:443`, `embedding = embedding / dd`), where
        // `degree[i] = Σ_j W[i,j]`. With the RBF diagonal at 0 this off-diagonal
        // row-sum matches scipy's `csgraph_laplacian` degree exactly.
        let mut dd = vec![0.0; n];
        for i in 0..n {
            let degree: f64 = (0..n).map(|j| w[[i, j]]).sum();
            dd[i] = degree.sqrt();
        }

        // Skip the first (trivial, eigenvalue ~ 0) eigenvector, take next n_components.
        let n_comp = self.n_components;
        let mut embedding = Array2::<f64>::zeros((n, n_comp));
        for (k, &idx) in indices.iter().skip(1).take(n_comp).enumerate() {
            for i in 0..n {
                let v = eigenvectors[[i, idx]];
                // Divide by `dd` of the row (guard against zero degree, mirroring
                // the `deg > 1e-15` guard in `normalised_laplacian`).
                embedding[[i, k]] = if dd[i] > 1e-15 { v / dd[i] } else { v };
            }
        }

        // sklearn applies `_deterministic_vector_sign_flip`
        // (`_spectral_embedding.py:465`) AFTER the `/dd` rescale: for each column,
        // find the row with the maximum absolute value; if that entry is negative,
        // negate the entire column.
        for k in 0..n_comp {
            let mut max_abs = 0.0;
            let mut sign = 1.0;
            for i in 0..n {
                let v = embedding[[i, k]];
                if v.abs() > max_abs {
                    max_abs = v.abs();
                    sign = if v < 0.0 { -1.0 } else { 1.0 };
                }
            }
            if sign < 0.0 {
                for i in 0..n {
                    embedding[[i, k]] = -embedding[[i, k]];
                }
            }
        }

        Ok(FittedSpectralEmbedding {
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

    /// Helper: two well-separated clusters.
    fn two_clusters() -> Array2<f64> {
        array![
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],
            [5.0, 5.0],
            [5.1, 5.0],
            [5.0, 5.1],
            [5.1, 5.1],
        ]
    }

    /// Helper: simple dataset.
    fn simple_data() -> Array2<f64> {
        array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0],]
    }

    #[test]
    fn test_spectral_embedding_basic_shape() {
        let se = SpectralEmbedding::new(2);
        let x = two_clusters();
        let fitted = se.fit(&x, &()).unwrap();
        assert_eq!(fitted.embedding().dim(), (8, 2));
    }

    #[test]
    fn test_spectral_embedding_1d() {
        let se = SpectralEmbedding::new(1);
        let x = two_clusters();
        let fitted = se.fit(&x, &()).unwrap();
        assert_eq!(fitted.embedding().ncols(), 1);
    }

    #[test]
    fn test_spectral_embedding_rbf_separates_clusters() {
        let se = SpectralEmbedding::new(1).with_affinity(Affinity::RBF { gamma: 1.0 });
        let x = two_clusters();
        let fitted = se.fit(&x, &()).unwrap();
        let emb = fitted.embedding();

        // The first 4 points (cluster 1) should have similar values,
        // and the last 4 (cluster 2) should have similar values, with
        // a gap between them.
        let c1_mean: f64 = (0..4).map(|i| emb[[i, 0]]).sum::<f64>() / 4.0;
        let c2_mean: f64 = (4..8).map(|i| emb[[i, 0]]).sum::<f64>() / 4.0;
        assert!(
            (c1_mean - c2_mean).abs() > 0.01,
            "clusters should be separated: c1={c1_mean}, c2={c2_mean}"
        );
    }

    #[test]
    fn test_spectral_embedding_knn_affinity() {
        let se =
            SpectralEmbedding::new(2).with_affinity(Affinity::NearestNeighbors { n_neighbors: 3 });
        let x = two_clusters();
        let fitted = se.fit(&x, &()).unwrap();
        assert_eq!(fitted.embedding().dim(), (8, 2));
    }

    #[test]
    fn test_spectral_embedding_invalid_n_components_zero() {
        let se = SpectralEmbedding::new(0);
        let x = simple_data();
        assert!(se.fit(&x, &()).is_err());
    }

    #[test]
    fn test_spectral_embedding_n_components_too_large() {
        let se = SpectralEmbedding::new(5); // n_samples = 5, need < 5
        let x = simple_data();
        assert!(se.fit(&x, &()).is_err());
    }

    #[test]
    fn test_spectral_embedding_insufficient_samples() {
        let se = SpectralEmbedding::new(1);
        let x = array![[1.0, 2.0]]; // 1 sample
        assert!(se.fit(&x, &()).is_err());
    }

    #[test]
    fn test_spectral_embedding_knn_n_neighbors_zero() {
        let se =
            SpectralEmbedding::new(1).with_affinity(Affinity::NearestNeighbors { n_neighbors: 0 });
        let x = simple_data();
        assert!(se.fit(&x, &()).is_err());
    }

    #[test]
    fn test_spectral_embedding_getters() {
        let se = SpectralEmbedding::new(3).with_affinity(Affinity::RBF { gamma: 0.5 });
        assert_eq!(se.n_components(), 3);
        assert_eq!(se.affinity(), Affinity::RBF { gamma: 0.5 });
    }

    #[test]
    fn test_spectral_embedding_knn_too_many_neighbors() {
        let se = SpectralEmbedding::new(1)
            .with_affinity(Affinity::NearestNeighbors { n_neighbors: 100 });
        let x = simple_data(); // 5 samples
        assert!(se.fit(&x, &()).is_err());
    }

    #[test]
    fn test_spectral_embedding_negative_gamma() {
        let se = SpectralEmbedding::new(1).with_affinity(Affinity::RBF { gamma: -1.0 });
        let x = simple_data();
        assert!(se.fit(&x, &()).is_err());
    }

    #[test]
    fn test_spectral_embedding_larger_dataset() {
        let n = 20;
        let d = 3;
        let mut data = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            for j in 0..d {
                data[[i, j]] = (i * d + j) as f64 / (n * d) as f64;
            }
        }
        let se = SpectralEmbedding::new(2);
        let fitted = se.fit(&data, &()).unwrap();
        assert_eq!(fitted.embedding().dim(), (20, 2));
    }
}
