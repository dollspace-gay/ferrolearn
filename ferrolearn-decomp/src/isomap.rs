//! Isomap (Isometric Mapping).
//!
//! Non-linear dimensionality reduction that preserves geodesic (shortest-path)
//! distances along the data manifold.
//!
//! # Algorithm
//!
//! 1. Build a k-nearest-neighbor graph over the data.
//! 2. Compute shortest paths between all pairs of points using Dijkstra's
//!    algorithm.
//! 3. Apply classical MDS to the geodesic distance matrix to obtain the
//!    low-dimensional embedding.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_decomp::Isomap;
//! use ferrolearn_core::traits::{Fit, Transform};
//! use ndarray::array;
//!
//! let iso = Isomap::new(2).with_n_neighbors(3);
//! let x = array![
//!     [0.0, 0.0],
//!     [1.0, 0.0],
//!     [2.0, 0.0],
//!     [0.0, 1.0],
//!     [1.0, 1.0],
//!     [2.0, 1.0],
//!     [0.0, 2.0],
//!     [1.0, 2.0],
//!     [2.0, 2.0],
//! ];
//! let fitted = iso.fit(&x, &()).unwrap();
//! let emb = fitted.embedding();
//! assert_eq!(emb.ncols(), 2);
//! ```
//!
//! ## REQ status
//!
//! Translation target: scikit-learn 1.5.2 `class Isomap`
//! (`sklearn/manifold/_isomap.py` — `KernelPCA` on the centered geodesic kernel).
//! Tracking: #1467. Each REQ is BINARY — SHIPPED (impl + non-test consumer +
//! tests + green verification) or NOT-STARTED (with a concrete open blocker).
//!
//! | REQ | Scope | Status | Evidence / Blocker |
//! |-----|-------|--------|--------------------|
//! | REQ-1 | Isomap embedding VALUE parity (classical MDS on geodesic distances + KernelPCA `svd_flip` deterministic sign) | SHIPPED | `fit` applies the per-column max-abs-positive sign flip (sklearn `svd_flip` `_kernel_pca.py:373`); element-wise matches live sklearn EXACTLY (no sign alignment) across n_neighbors {3,4,5}, n_components {1,2,3}, 3 fixtures in `tests/divergence_isomap.rs` (was #1468, fixed). Consumer: re-export `lib.rs:89` |
//! | REQ-2 | Geodesic distance matrix parity (kNN graph + Dijkstra == sklearn `dist_matrix_`) | SHIPPED | `build_knn_graph` + `all_pairs_shortest_paths` match sklearn `kneighbors_graph(mode=distance)` + `shortest_path` (`_isomap.py:242-299`); verified via sign-robust embedding + embedding-distance equality |
//! | REQ-3 | Structural (embedding shape, deterministic) | SHIPPED (scoped) | shape + determinism guards. NOTE disconnected-graph: ferrolearn errors (NumericalInstability), sklearn warns+completes (REQ-9) |
//! | REQ-4 | Error/parameter contracts (n_components 0/>n, n_neighbors 0/≥n, <2 samples, disconnected, NON-FINITE rejection) | SHIPPED (scoped) | `fit`/`transform` guards; divergence error tests. NON-FINITE: `fit` (BEFORE the kNN graph) + `transform` call `reject_non_finite` (`isomap.rs` symbol `reject_non_finite`), returning the CLEAN finiteness `InvalidParameter{name:"X", reason:"Input X contains NaN or infinity."}` = sklearn `nbrs_.fit/kneighbors` `_validate_data(force_all_finite=True)` (`_isomap.py:228`,`:411`,`utils/validation.py:147-154`) — REPLACES the incidental "kNN graph disconnected" `NumericalInstability` for non-finite X (R-DEV-2). `tests/divergence_nonfinite_spillover.rs::divergence_isomap_fit_nan`/`_transform_nan` match the live sklearn 1.5.2 oracle (#2290) |
//! | REQ-5 | `transform` out-of-sample (geodesic-graph-linked, not Euclidean) | SHIPPED | `transform` links each query into the TRAINING geodesic graph: kNN of the query (Euclidean, same `k`) → `G_X[j] = min_k(dist_matrix_[k,j] + d_k)` (sklearn `_isomap.py:430`) → `K_pred = -0.5*G_X²` (`:432-433`) → out-of-sample `KernelCenterer` double-centering with the TRAINING `G = -0.5*dist_matrix_²` means (`preprocessing/_data.py:2455-2459`) → project onto `scaled_alphas = eigenvectors_/sqrt(eigenvalues_) = embedding_/eigenvalues_` (`_kernel_pca.py:504-512`), reusing the sign-flipped `embedding_` so the sign matches REQ-1. Stored fit state: `dist_matrix_`, `g_fit_col_means_`/`g_fit_grand_mean_` (the `G` centerer means), `eigenvalues_`, `embedding_`, `n_neighbors_`. Matches the live sklearn 1.5.2 oracle to <1e-6 sign-exact (`tests/divergence_isomap_transform_2401.rs::divergence_transform_out_of_sample`, was #1469/#2401). Consumer: re-export `lib.rs:89` |
//! | REQ-6 | `radius` mode + `radius_neighbors_graph` | NOT-STARTED | sklearn `_isomap.py:253-261` — blocker #1470 |
//! | REQ-7 | `path_method` (FW/Floyd-Warshall vs D/Dijkstra) + `eigen_solver`/`tol`/`max_iter` | NOT-STARTED | sklearn `_isomap.py:299,233-240` — blocker #1471 |
//! | REQ-8 | `metric`/`p` (non-Euclidean minkowski) + `metric_params` | NOT-STARTED | Euclidean only; sklearn `_isomap.py:194-195` — blocker #1472 |
//! | REQ-9 | `reconstruction_error()`/`dist_matrix_`/`kernel_pca_`/`nbrs_` attrs + `neighbors_algorithm` + disconnected-graph completion + degenerate-eigenvalue subspace (CARVE-OUT) | NOT-STARTED | sklearn `_isomap.py:267-335` — blocker #1473 |
//! | REQ-10 | PyO3 binding | NOT-STARTED | no `ferrolearn-python` registration — blocker #1474 |
//! | REQ-11 | ferray substrate | NOT-STARTED | dense `Array2` only — blocker #1475 |

use crate::mds::{classical_mds, eigh_faer, pairwise_sq_distances};
use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Transform};
use ndarray::Array2;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Reject non-finite input the way sklearn's `_validate_data` does.
///
/// sklearn validates X with `force_all_finite=True` (the `check_array` default)
/// inside `Isomap`'s `nbrs_.fit(X)` / `nbrs_.kneighbors(X)` (NearestNeighbors)
/// at the top of `_fit_transform` (`sklearn/manifold/_isomap.py:228`) and
/// `transform` (`:411`), raising `ValueError("Input X contains NaN.")` /
/// `"... contains infinity ..."` (`sklearn/utils/validation.py:147-154`) BEFORE
/// the kNN graph / geodesic / MDS math. Calling this FIRST means a non-finite X
/// yields the CLEAN finiteness rejection instead of the incidental "kNN graph
/// disconnected" `NumericalInstability`. NaN AND infinity are both rejected. The
/// message names "NaN" and "infinity" to mirror sklearn's `ValueError`. Never
/// panics (R-CODE-2).
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
// Isomap (unfitted)
// ---------------------------------------------------------------------------

/// Isomap configuration.
///
/// Holds hyperparameters for the Isomap algorithm. Call [`Fit::fit`] to compute
/// the geodesic embedding and obtain a [`FittedIsomap`].
#[derive(Debug, Clone)]
pub struct Isomap {
    /// Number of embedding dimensions.
    n_components: usize,
    /// Number of nearest neighbors for the kNN graph.
    n_neighbors: usize,
}

impl Isomap {
    /// Create a new `Isomap` with `n_components` embedding dimensions.
    ///
    /// The default number of neighbors is 5.
    #[must_use]
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            n_neighbors: 5,
        }
    }

    /// Set the number of nearest neighbors.
    #[must_use]
    pub fn with_n_neighbors(mut self, k: usize) -> Self {
        self.n_neighbors = k;
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
}

// ---------------------------------------------------------------------------
// FittedIsomap
// ---------------------------------------------------------------------------

/// A fitted Isomap model holding the learned embedding and training data.
///
/// Created by calling [`Fit::fit`] on an [`Isomap`]. Implements
/// [`Transform<Array2<f64>>`] for out-of-sample projection by linking each query
/// into the training geodesic graph and applying the precomputed-KernelPCA
/// projection — mirroring sklearn `Isomap.transform`
/// (`sklearn/manifold/_isomap.py:386-435`).
#[derive(Debug, Clone)]
pub struct FittedIsomap {
    /// The embedding, shape `(n_samples, n_components)`. Carries the deterministic
    /// `svd_flip` sign (REQ-1, clean). The transform reuses this exact array so
    /// its sign is consistent with the embedding by construction.
    embedding_: Array2<f64>,
    /// Training data, stored for the query's k-nearest-training-neighbor search.
    x_train_: Array2<f64>,
    /// Number of neighbors used during fitting (the query kNN uses the same `k`,
    /// matching sklearn `self.nbrs_.kneighbors(X)`, `_isomap.py:411`).
    n_neighbors_: usize,
    /// Kernel matrix eigenvalues from the MDS step (top n_components, clamped
    /// `>= 0`). Equal to sklearn's `kernel_pca_.eigenvalues_`.
    eigenvalues_: Vec<f64>,
    /// Geodesic shortest-path distance matrix, shape `(n_train, n_train)`. Equal
    /// to sklearn's `dist_matrix_` (`_isomap.py:299`); the transform links each
    /// query through it (`G_X[i] = min_k(dist_matrix_[k] + dist(query, k))`,
    /// `_isomap.py:430`).
    dist_matrix_: Array2<f64>,
    /// Column means of the TRAINING precomputed kernel `G = -0.5 * dist_matrix_²`
    /// (sklearn `KernelCenterer.K_fit_rows_`, `preprocessing/_data.py:2423`). Since
    /// `dist_matrix_` is symmetric, `G`'s column means equal its row means; this is
    /// `-0.5 * mean_j(dist_matrix_[i,j]²)` per index.
    g_fit_col_means_: Vec<f64>,
    /// Grand mean of the TRAINING precomputed kernel `G = -0.5 * dist_matrix_²`
    /// (sklearn `KernelCenterer.K_fit_all_`, `preprocessing/_data.py:2424`).
    g_fit_grand_mean_: f64,
}

impl FittedIsomap {
    /// The embedding coordinates, shape `(n_samples, n_components)`.
    #[must_use]
    pub fn embedding(&self) -> &Array2<f64> {
        &self.embedding_
    }
}

// ---------------------------------------------------------------------------
// kNN + Dijkstra helpers
// ---------------------------------------------------------------------------

/// A state for Dijkstra's priority queue.
#[derive(Clone, PartialEq)]
struct State {
    cost: f64,
    node: usize,
}

impl Eq for State {}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        // Flip for min-heap
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Build a k-nearest-neighbor adjacency list from squared distances.
/// Returns `adj[i]` = Vec of (neighbor_index, distance).
fn build_knn_graph(sq_dist: &Array2<f64>, k: usize) -> Vec<Vec<(usize, f64)>> {
    let n = sq_dist.nrows();
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];

    for i in 0..n {
        // Collect (distance, index) for all other points.
        let mut neighbors: Vec<(f64, usize)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (sq_dist[[i, j]].sqrt(), j))
            .collect();
        neighbors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        for &(dist, j) in neighbors.iter().take(k) {
            adj[i].push((j, dist));
        }
    }

    // Make the graph symmetric: if i is a neighbor of j, j is a neighbor of i.
    let mut sym: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for i in 0..n {
        for &(j, d) in &adj[i] {
            sym[i].push((j, d));
            sym[j].push((i, d));
        }
    }
    // Deduplicate keeping shortest distance.
    for entry in &mut sym {
        entry.sort_by_key(|a| a.0);
        entry.dedup_by(|a, b| {
            if a.0 == b.0 {
                b.1 = b.1.min(a.1);
                true
            } else {
                false
            }
        });
    }
    sym
}

/// Dijkstra shortest path from a single source.
fn dijkstra(adj: &[Vec<(usize, f64)>], source: usize) -> Vec<f64> {
    let n = adj.len();
    let mut dist = vec![f64::INFINITY; n];
    dist[source] = 0.0;
    let mut heap = BinaryHeap::new();
    heap.push(State {
        cost: 0.0,
        node: source,
    });

    while let Some(State { cost, node }) = heap.pop() {
        if cost > dist[node] {
            continue;
        }
        for &(neighbor, weight) in &adj[node] {
            let next_cost = cost + weight;
            if next_cost < dist[neighbor] {
                dist[neighbor] = next_cost;
                heap.push(State {
                    cost: next_cost,
                    node: neighbor,
                });
            }
        }
    }
    dist
}

/// Compute all-pairs shortest paths via Dijkstra.
fn all_pairs_shortest_paths(adj: &[Vec<(usize, f64)>]) -> Array2<f64> {
    let n = adj.len();
    let mut result = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        let dists = dijkstra(adj, i);
        for j in 0..n {
            result[[i, j]] = dists[j];
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl Fit<Array2<f64>, ()> for Isomap {
    type Fitted = FittedIsomap;
    type Error = FerroError;

    /// Fit Isomap by building the kNN graph, computing geodesic distances,
    /// and applying classical MDS.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `n_components` is zero,
    ///   `n_neighbors` is zero, or `n_components > n_samples`.
    /// - [`FerroError::InsufficientSamples`] if there are fewer than 2 samples
    ///   or fewer samples than `n_neighbors + 1`.
    /// - [`FerroError::NumericalInstability`] if the kNN graph is disconnected.
    fn fit(&self, x: &Array2<f64>, _y: &()) -> Result<FittedIsomap, FerroError> {
        let n_samples = x.nrows();

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
        if n_samples < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: n_samples,
                context: "Isomap::fit requires at least 2 samples".into(),
            });
        }
        if self.n_neighbors >= n_samples {
            return Err(FerroError::InvalidParameter {
                name: "n_neighbors".into(),
                reason: format!(
                    "n_neighbors ({}) must be less than n_samples ({})",
                    self.n_neighbors, n_samples
                ),
            });
        }
        if self.n_components > n_samples {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: format!(
                    "n_components ({}) exceeds n_samples ({})",
                    self.n_components, n_samples
                ),
            });
        }

        // Reject NaN/Inf BEFORE the kNN graph, so a non-finite X gives the CLEAN
        // finiteness rejection rather than the incidental "kNN graph
        // disconnected" `NumericalInstability` (sklearn validates X inside
        // `nbrs_.fit(X)` with `force_all_finite=True`, `_isomap.py:228`,
        // `utils/validation.py:147-154`).
        reject_non_finite(x)?;

        // Step 1: pairwise Euclidean distances.
        let sq_dist = pairwise_sq_distances(x);

        // Step 2: Build kNN graph.
        let adj = build_knn_graph(&sq_dist, self.n_neighbors);

        // Step 3: All-pairs shortest paths.
        let geodesic = all_pairs_shortest_paths(&adj);

        // Check for disconnected graph.
        for i in 0..n_samples {
            for j in 0..n_samples {
                if geodesic[[i, j]].is_infinite() {
                    return Err(FerroError::NumericalInstability {
                        message: format!(
                            "kNN graph is disconnected (no path from point {i} to {j}). \
                             Try increasing n_neighbors."
                        ),
                    });
                }
            }
        }

        // Step 4: Classical MDS on the geodesic distance matrix.
        let geo_sq = geodesic.mapv(|v| v * v);

        // We need extra info for Nystroem extension, so we do the MDS manually.
        let n = n_samples;
        let n_f = n as f64;
        let mut row_means = vec![0.0; n];
        let mut grand_mean = 0.0;
        for i in 0..n {
            for j in 0..n {
                row_means[i] += geo_sq[[i, j]];
                grand_mean += geo_sq[[i, j]];
            }
            row_means[i] /= n_f;
        }
        grand_mean /= n_f * n_f;

        let (mut embedding, _stress) = classical_mds(&geo_sq, self.n_components)?;

        // Apply KernelPCA's `svd_flip` sign convention for deterministic signs
        // (`sklearn/decomposition/_kernel_pca.py:373`, rule at
        // `sklearn/utils/extmath.py:888-896`): for each embedding column, find
        // the row of maximum absolute value (first occurrence on ties, like
        // numpy `argmax`) and flip the column's sign so that entry is positive.
        // `classical_mds` scales each eigenvector by `sqrt(lambda_k) >= 0`, a
        // positive per-column constant, so `argmax(|scaled|)` equals
        // `argmax(|eigenvector|)` and this matches svd_flip on the unit vector.
        for k in 0..embedding.ncols() {
            let mut max_abs = 0.0_f64;
            let mut neg = false;
            for i in 0..n {
                let v = embedding[[i, k]];
                let a = v.abs();
                if a > max_abs {
                    max_abs = a;
                    neg = v < 0.0;
                }
            }
            if neg {
                for i in 0..n {
                    embedding[[i, k]] = -embedding[[i, k]];
                }
            }
        }

        // Eigendecompose for Nystroem extension storage
        let mut b = Array2::<f64>::zeros((n, n));
        let mut col_means = vec![0.0; n];
        for j in 0..n {
            for i in 0..n {
                col_means[j] += geo_sq[[i, j]];
            }
            col_means[j] /= n_f;
        }
        for i in 0..n {
            for j in 0..n {
                b[[i, j]] = -0.5 * (geo_sq[[i, j]] - row_means[i] - col_means[j] + grand_mean);
            }
        }

        let (eigenvalues, _eigenvectors) = eigh_faer(&b)?;

        // Sort eigenvalues descending, select top n_components. The eigenVALUES
        // alone are needed: the transform projects with the scaled alphas
        // `eigenvectors_ / sqrt(eigenvalues_)`, and since
        // `embedding_[:,k] = eigenvectors_[:,k] * sqrt(eigenvalues_[k])`
        // (`mds.rs` `embedding[i,k] = v_ik * sqrt(lambda_k)`), the scaled alphas
        // are `embedding_[:,k] / eigenvalues_[k]` — derivable from the
        // SIGN-FLIPPED `embedding`, so the transform sign is consistent with the
        // embedding (REQ-1) by construction, no separate sign flip needed.
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b_idx| {
            eigenvalues[b_idx]
                .partial_cmp(&eigenvalues[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let n_comp = self.n_components.min(n);
        let mut top_eigenvalues = Vec::with_capacity(n_comp);
        for &idx in indices.iter().take(n_comp) {
            top_eigenvalues.push(eigenvalues[idx].max(0.0));
        }

        // The TRAINING precomputed kernel is `G = -0.5 * dist_matrix_²`
        // (sklearn `_isomap.py:306-307`); `kernel_pca_`'s `KernelCenterer` was fit
        // on it. Its column means (`K_fit_rows_`) and grand mean (`K_fit_all_`,
        // `preprocessing/_data.py:2423-2424`) are `-0.5 *` the `geo_sq` means
        // computed above (`geo_sq = dist_matrix_²`, symmetric ⇒ col mean = row
        // mean). Store them so the transform can double-center `G_X` with the
        // TRAINING means (out-of-sample KernelCenterer, `_data.py:2455-2459`).
        let g_fit_col_means_: Vec<f64> = col_means.iter().map(|&m| -0.5 * m).collect();
        let g_fit_grand_mean_ = -0.5 * grand_mean;

        Ok(FittedIsomap {
            embedding_: embedding,
            x_train_: x.to_owned(),
            n_neighbors_: self.n_neighbors,
            eigenvalues_: top_eigenvalues,
            dist_matrix_: geodesic,
            g_fit_col_means_,
            g_fit_grand_mean_,
        })
    }
}

impl Transform<Array2<f64>> for FittedIsomap {
    type Output = Array2<f64>;
    type Error = FerroError;

    /// Project new data into the Isomap embedding space.
    ///
    /// Mirrors sklearn `Isomap.transform` (`sklearn/manifold/_isomap.py:386-435`):
    /// link each query INTO the training geodesic graph, then apply the
    /// precomputed-KernelPCA projection.
    ///
    /// For each query `q`:
    /// 1. find its `n_neighbors` nearest TRAINING points (Euclidean), with their
    ///    distances `d_k` and training indices `k` (sklearn
    ///    `nbrs_.kneighbors(X)`, `_isomap.py:411`);
    /// 2. the geodesic distance to every training point `j` is the shortest path
    ///    THROUGH the graph:
    ///    `G_X[j] = min_k (dist_matrix_[k, j] + d_k)` (`_isomap.py:430`);
    /// 3. form the precomputed kernel row `K_pred = -0.5 * G_X²`
    ///    (`_isomap.py:432-433`);
    /// 4. double-center `K_pred` with the TRAINING `KernelCenterer` means
    ///    (`K_pred[j] -= K_fit_rows_[j]; K_pred[j] -= rowmean(K_pred);
    ///    K_pred += K_fit_all_`, `preprocessing/_data.py:2455-2459`);
    /// 5. project onto the scaled eigenvectors `eigenvectors_ / sqrt(eigenvalues_)`
    ///    (`_kernel_pca.py:504-512`). Since
    ///    `embedding_[:,c] = eigenvectors_[:,c] * sqrt(eigenvalues_[c])`, the
    ///    scaled alphas equal `embedding_[:,c] / eigenvalues_[c]` — derived from
    ///    the sign-flipped `embedding`, so the transform sign is consistent with
    ///    the embedding (REQ-1) by construction.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] if the number of features does not match
    ///   the training data.
    /// - [`FerroError::InvalidParameter`] if `x` contains NaN or infinity.
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>, FerroError> {
        let n_features = self.x_train_.ncols();
        if x.ncols() != n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedIsomap::transform".into(),
            });
        }

        // Reject NaN/Inf BEFORE the kNN / projection (sklearn validates X inside
        // `nbrs_.kneighbors(X)` with `force_all_finite=True`, `_isomap.py:411`,
        // `utils/validation.py:147-154`).
        reject_non_finite(x)?;

        let n_test = x.nrows();
        let n_train = self.x_train_.nrows();
        let n_comp = self.eigenvalues_.len();
        let k = self.n_neighbors_;

        let mut result = Array2::<f64>::zeros((n_test, n_comp));

        for t in 0..n_test {
            // Step 1: the query's `n_neighbors` nearest TRAINING points (Euclidean),
            // keeping each neighbor's distance `d_k` and training index `k`
            // (sklearn `nbrs_.kneighbors(X)`, `_isomap.py:411`).
            let mut dists: Vec<(f64, usize)> = (0..n_train)
                .map(|i| {
                    let mut sq = 0.0;
                    for f in 0..n_features {
                        let diff = x[[t, f]] - self.x_train_[[i, f]];
                        sq += diff * diff;
                    }
                    (sq.sqrt(), i)
                })
                .collect();
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
            let neighbors = &dists[..k.min(n_train)];

            // Step 2: geodesic distance from the query to every training point `j`
            // is the shortest path through the graph via one of the query's kNN:
            // `G_X[j] = min_k (dist_matrix_[k, j] + d_k)` (`_isomap.py:430`).
            // Step 3: precomputed kernel row `K_pred[j] = -0.5 * G_X[j]²`
            // (`_isomap.py:432-433`).
            let mut k_pred = vec![0.0_f64; n_train];
            for (j, slot) in k_pred.iter_mut().enumerate() {
                let mut g_xj = f64::INFINITY;
                for &(d_k, nbr) in neighbors {
                    let cand = self.dist_matrix_[[nbr, j]] + d_k;
                    if cand < g_xj {
                        g_xj = cand;
                    }
                }
                *slot = -0.5 * g_xj * g_xj;
            }

            // Step 4: out-of-sample KernelCenterer double-centering with the
            // TRAINING means (`preprocessing/_data.py:2455-2459`):
            // `K_pred_cols = rowmean(K_pred)` over the n_train training columns,
            // then `K_pred[j] -= K_fit_rows_[j]; K_pred[j] -= K_pred_cols;
            // K_pred += K_fit_all_`.
            let row_mean: f64 = k_pred.iter().sum::<f64>() / n_train as f64;
            for (j, val) in k_pred.iter_mut().enumerate() {
                *val = *val - self.g_fit_col_means_[j] - row_mean + self.g_fit_grand_mean_;
            }

            // Step 5: project onto the scaled eigenvectors
            // `scaled_alphas[:,c] = eigenvectors_[:,c] / sqrt(eigenvalues_[c])`
            // (`_kernel_pca.py:504-512`). With
            // `embedding_[i,c] = eigenvectors_[i,c] * sqrt(eigenvalues_[c])`, this
            // is `embedding_[i,c] / eigenvalues_[c]` — reusing the sign-flipped
            // `embedding`, so the sign matches REQ-1. Null-space components
            // (eigenvalue 0) contribute nothing, matching sklearn's
            // `non_zeros = np.flatnonzero(self.eigenvalues_)` masking
            // (`_kernel_pca.py:505-509`).
            for c in 0..n_comp {
                let eigval = self.eigenvalues_[c];
                if eigval <= 0.0 {
                    continue;
                }
                let mut sum = 0.0;
                for (j, &kpj) in k_pred.iter().enumerate() {
                    sum += kpj * (self.embedding_[[j, c]] / eigval);
                }
                result[[t, c]] = sum;
            }
        }

        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Helper: simple line dataset.
    fn line_data() -> Array2<f64> {
        array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0],]
    }

    /// Helper: 2D grid.
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

    #[test]
    fn test_isomap_basic_shape() {
        let iso = Isomap::new(2).with_n_neighbors(3);
        let x = grid_data();
        let fitted = iso.fit(&x, &()).unwrap();
        assert_eq!(fitted.embedding().dim(), (9, 2));
    }

    #[test]
    fn test_isomap_1d() {
        let iso = Isomap::new(1).with_n_neighbors(2);
        let x = line_data();
        let fitted = iso.fit(&x, &()).unwrap();
        assert_eq!(fitted.embedding().ncols(), 1);
    }

    #[test]
    fn test_isomap_preserves_ordering() {
        // Points on a line: the embedding should preserve the ordering.
        let iso = Isomap::new(1).with_n_neighbors(2);
        let x = line_data();
        let fitted = iso.fit(&x, &()).unwrap();
        let emb = fitted.embedding();
        let vals: Vec<f64> = (0..5).map(|i| emb[[i, 0]]).collect();
        // Check that values are monotonic (up to sign).
        let ascending = vals.windows(2).all(|w| w[0] <= w[1] + 1e-10);
        let descending = vals.windows(2).all(|w| w[0] >= w[1] - 1e-10);
        assert!(
            ascending || descending,
            "embedding should be monotonic: {vals:?}"
        );
    }

    #[test]
    fn test_isomap_transform_new_data() {
        let iso = Isomap::new(2).with_n_neighbors(3);
        let x_train = grid_data();
        let fitted = iso.fit(&x_train, &()).unwrap();
        let x_test = array![[0.5, 0.5], [1.5, 1.5]];
        let projected = fitted.transform(&x_test).unwrap();
        assert_eq!(projected.dim(), (2, 2));
    }

    #[test]
    fn test_isomap_transform_shape_mismatch() {
        let iso = Isomap::new(2).with_n_neighbors(3);
        let x = grid_data();
        let fitted = iso.fit(&x, &()).unwrap();
        let x_bad = array![[1.0, 2.0, 3.0]]; // 3 features instead of 2
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_isomap_transform_recovers_training() {
        // Transforming the training data should produce something close to
        // the stored embedding.
        let iso = Isomap::new(2).with_n_neighbors(3);
        let x = grid_data();
        let fitted = iso.fit(&x, &()).unwrap();
        let projected = fitted.transform(&x).unwrap();
        let emb = fitted.embedding();
        // Transforming the training data reproduces the stored embedding: for a
        // training point, its kNN includes itself at distance 0, so
        // `G_X[j] = min_k(dist_matrix_[k,j] + d_k) = dist_matrix_[self, j]`, and
        // the precomputed-KernelPCA projection equals the embedding row (sklearn
        // `transform(X_train) == embedding_`). Sign-EXACT (the transform reuses
        // the sign-flipped `embedding_` as its scaled alphas).
        assert_eq!(projected.dim(), emb.dim());
        for i in 0..emb.nrows() {
            for j in 0..emb.ncols() {
                assert!(
                    (projected[[i, j]] - emb[[i, j]]).abs() < 1e-9,
                    "transform(X_train) must reproduce embedding_ at [{i},{j}]"
                );
            }
        }
    }

    #[test]
    fn test_isomap_invalid_n_components_zero() {
        let iso = Isomap::new(0);
        let x = grid_data();
        assert!(iso.fit(&x, &()).is_err());
    }

    #[test]
    fn test_isomap_invalid_n_neighbors_zero() {
        let iso = Isomap::new(2).with_n_neighbors(0);
        let x = grid_data();
        assert!(iso.fit(&x, &()).is_err());
    }

    #[test]
    fn test_isomap_n_neighbors_too_large() {
        let iso = Isomap::new(2).with_n_neighbors(100);
        let x = grid_data(); // 9 samples
        assert!(iso.fit(&x, &()).is_err());
    }

    #[test]
    fn test_isomap_insufficient_samples() {
        let iso = Isomap::new(1).with_n_neighbors(1);
        let x = array![[1.0, 2.0]]; // 1 sample
        assert!(iso.fit(&x, &()).is_err());
    }

    #[test]
    fn test_isomap_getters() {
        let iso = Isomap::new(3).with_n_neighbors(7);
        assert_eq!(iso.n_components(), 3);
        assert_eq!(iso.n_neighbors(), 7);
    }

    #[test]
    fn test_isomap_default_n_neighbors() {
        let iso = Isomap::new(2);
        assert_eq!(iso.n_neighbors(), 5);
    }

    #[test]
    fn test_isomap_n_components_too_large() {
        let iso = Isomap::new(50);
        let x = grid_data(); // 9 samples
        assert!(iso.fit(&x, &()).is_err());
    }
}
