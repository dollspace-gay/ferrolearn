//! Agglomerative (bottom-up) hierarchical clustering.
//!
//! This module provides [`AgglomerativeClustering`], a hierarchical clustering
//! algorithm that builds a dendrogram by successively merging the two closest
//! clusters.  The merge criterion is determined by the [`Linkage`] strategy.
//!
//! # Algorithm
//!
//! 1. Initialise each data point as its own singleton cluster.
//! 2. Build an `n × n` pairwise distance matrix.
//! 3. Repeat until `n_clusters` clusters remain:
//!    a. Find the pair of clusters `(i, j)` with the smallest inter-cluster
//!    distance according to the chosen linkage.
//!    b. Merge them into a new cluster; record the merge in `children_`.
//!    c. Update distances using the Lance–Williams recurrence.
//!
//! The overall complexity is **O(n³)** in time and **O(n²)** in space, which
//! is practical for datasets up to a few thousand samples.
//!
//! # Linkage strategies
//!
//! | [`Linkage`]  | Distance formula | Properties |
//! |--------------|------------------|------------|
//! | `Single`     | `min d(a, b)`    | Chaining effect; handles non-convex shapes |
//! | `Complete`   | `max d(a, b)`    | Compact clusters |
//! | `Average`    | mean of pairwise | Compromise |
//! | `Ward`       | increase in SSE  | Minimises within-cluster variance |
//!
//! # Note
//!
//! [`AgglomerativeClustering`] implements [`Fit`] only.  There is no
//! `predict` method (mirroring scikit-learn's design).
//!
//! # Examples
//!
//! ```
//! use ferrolearn_cluster::AgglomerativeClustering;
//! use ferrolearn_core::Fit;
//! use ndarray::Array2;
//!
//! let x = Array2::from_shape_vec((6, 2), vec![
//!     0.0, 0.0,  0.1, 0.0,  0.0, 0.1,
//!     8.0, 8.0,  8.1, 8.0,  8.0, 8.1,
//! ]).unwrap();
//!
//! let model = AgglomerativeClustering::<f64>::new(2);
//! let fitted = model.fit(&x, &()).unwrap();
//! assert_eq!(fitted.labels().len(), 6);
//! ```
//!
//! # `## REQ status`
//!
//! Binary (R-DEFER-2), translating `sklearn/cluster/_agglomerative.py`
//! (`class AgglomerativeClustering(ClusterMixin, BaseEstimator)` `:781`).
//! Design doc: `.design/cluster/agglomerative.md`. Cites use ferrolearn symbol
//! anchors / sklearn `file:line` (commit 156ef14); expected values from the live
//! sklearn 1.5.2 oracle (R-CHAR-3). This is a verify-and-document unit: the
//! `labels_` PARTITION (up to a label permutation) genuinely ships on separable
//! data through real consumers, but the absolute `labels_` numbering and the full
//! `children_` dendrogram DIVERGE — both rooted in ferrolearn's truncated-tree +
//! ascending-slot relabel vs sklearn's full dendrogram + `_hc_cut` heap cut
//! (the shared #938 root cause, also gating `birch.rs` / `feature_agglomeration.rs`).
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`labels_` PARTITION up-to-permutation, separable data) | SHIPPED | `fn agglomerate` (Lance–Williams merge-until-`n_clusters` via `find_min_pair`/`pairwise_sq_dists`) → `Fit::fit` builds `labels_`, mirroring the merge clustering of sklearn `_fit` (`_agglomerative.py:992-1106`). Partition value-matches the oracle for all 4 linkages. Consumers: `_RsAgglomerativeClustering` (`ferrolearn-python/src/extras.rs`), `birch.rs fn fit`, `feature_agglomeration.rs fn fit`. Guards: `green_two_blobs_partition_all_linkages`, `green_three_blobs_partition_all_linkages` in `tests/divergence_agglomerative.rs`. |
//! | REQ-2 (`n_clusters_` == requested) | SHIPPED | `Fit::fit` sets `n_clusters_: self.n_clusters`, mirroring `self.n_clusters_ = self.n_clusters` when `distance_threshold is None` (`_agglomerative.py:1095`). Guards above assert `n_clusters_` 2 and 3; `n_clusters()` accessor consumed by `birch.rs`/`feature_agglomeration.rs`. |
//! | REQ-3 (four linkage criteria — partition) | SHIPPED | `enum Linkage` + `match linkage` arms in `fn agglomerate` (Single=min, Complete=max, Average=size-weighted mean, Ward=size-weighted Lance–Williams) mirror `_TREE_BUILDERS` (`_agglomerative.py:720-725`). Partition matches the oracle for all four (guards above). Caveat: PARTITION only — merge-distance VALUES/ties differ (squared-Euclidean LW vs sklearn heap/nn-chain), see REQ-9. |
//! | REQ-4 (`n_clusters=2` ctor default + sklearn error ABI) | NOT-STARTED | open prereq blocker #963. sklearn `__init__` defaults `n_clusters=2` (`_agglomerative.py:951`); ferrolearn `fn new(n_clusters)` requires it. Validation errors are `FerroError::InvalidParameter`/`InsufficientSamples` (crate-wide port convention), not sklearn's `ValueError`/`InvalidParameterError`. |
//! | REQ-5 (`ensure_min_samples=2` validation) | NOT-STARTED | open prereq blocker #964. sklearn `fit` → `_validate_data(X, ensure_min_samples=2)` (`_agglomerative.py:989`) rejects `n_samples < 2`; ferrolearn `fn fit` accepts a single sample when `n_clusters <= 1` (`test_single_sample_single_cluster`). Coupled fix: `birch.rs fn fit` calls `AgglomerativeClustering::new(1)` on a 1-row matrix in the single-subcluster path, so this is a multi-file change, not minimal in `agglomerative.rs` alone. |
//! | REQ-6 (`children_` full-dendrogram format) | NOT-STARTED | open prereq blocker #938. sklearn `children_` is shape `(n_samples-1, 2)` with internal-node IDs `>= n_samples` (`_agglomerative.py:902-908`); ferrolearn `FittedAgglomerativeClustering::children_` is length `n_samples - n_clusters` of reused merged-into-slot pairs (`fn agglomerate`: `children.push((ci, cj))`). Different length AND ID semantics — full-dendrogram rewrite, not minimal. |
//! | REQ-7 (`labels_` ABSOLUTE numbering via `_hc_cut`) | NOT-STARTED | open prereq blocker #938. sklearn numbers labels by a negated-id min-heap pop over the top-`n_clusters` dendrogram nodes (`_hc_cut`, `_agglomerative.py:760-775`); ferrolearn relabels by ascending surviving-slot order via a `HashMap` (`fn agglomerate` relabel loop). Same partition (REQ-1), permuted integers. Requires the full `children_` (REQ-6) then `_hc_cut`. |
//! | REQ-8 (`metric` / `connectivity`) | NOT-STARTED | open prereq blocker #965. sklearn `metric` ∈ {euclidean,l1,l2,manhattan,cosine,precomputed} default `'euclidean'` with the ward-requires-euclidean rule (`_agglomerative.py:795-799`, `:1034-1038`) and `connectivity` for structured clustering (`:812-822`). ferrolearn `fn sq_euclidean`/`fn pairwise_sq_dists` are Euclidean-only, unstructured. |
//! | REQ-9 (`distance_threshold`/`compute_full_tree`/`compute_distances`/`distances_`) | NOT-STARTED | open prereq blocker #966. sklearn `distance_threshold` (XOR with `n_clusters`, `_agglomerative.py:1022-1027`; `n_clusters_` derived `:1090-1093`), `compute_full_tree='auto'` (`:1051-1064`), `compute_distances` → `distances_` (`:1087-1088`). ferrolearn has only `n_clusters` + `linkage`, no `distances_`, and the merge-distance VALUES differ. |
//! | REQ-10 (`n_leaves_`/`n_connected_components_` + `memory`) | NOT-STARTED | open prereq blocker #967. sklearn sets `n_leaves_`/`n_connected_components_` from the tree builder (`_agglomerative.py:1083-1085`) and caches via `memory` (`:1006`/`:1076`). `FittedAgglomerativeClustering` exposes `labels()`/`n_clusters()`/`children()` only. |
//! | REQ-11 (PyO3 binding parity) | SHIPPED | `#[pyclass(name="_RsAgglomerativeClustering")]` (`ferrolearn-python/src/extras.rs`): `fn new(n_clusters=2)`, `fn fit` → `ferrolearn_cluster::AgglomerativeClustering::<f64>::new(self.n_clusters)`, `#[getter] labels_`; registered in `ferrolearn-python/src/lib.rs`, wrapped `class AgglomerativeClustering(_ClusterWrapper)` in `python/ferrolearn/_extras.py`, exported in `__init__.py`. `ferrolearn.AgglomerativeClustering(n_clusters=2).fit(X).labels_` matches sklearn up to label permutation (REQ-1). Hard-wires Ward (no `linkage`/`metric`/`distance_threshold` arg). |
//! | REQ-12 (ferray substrate) | NOT-STARTED | open prereq blocker #968. `agglomerative.rs` imports `ndarray::{Array1, Array2}` + `num_traits::Float`, not `ferray-core`; the PyO3 boundary uses `numpy2_to_ndarray`, not `ferray::numpy_interop` (R-SUBSTRATE). |

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::Fit;
use ndarray::{Array1, Array2};
use num_traits::Float;

// ─────────────────────────────────────────────────────────────────────────────
// Public enums & configuration struct
// ─────────────────────────────────────────────────────────────────────────────

/// The linkage criterion used to measure distances between clusters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Linkage {
    /// Ward linkage: merge the pair that minimises the increase in
    /// within-cluster sum of squared errors.
    Ward,
    /// Complete linkage: the distance between two clusters is the
    /// *maximum* distance between any pair of their members.
    Complete,
    /// Average linkage (UPGMA): the distance is the mean of all pairwise
    /// distances between the two clusters.
    Average,
    /// Single linkage: the distance between two clusters is the *minimum*
    /// pairwise distance between their members.
    Single,
}

/// Agglomerative clustering configuration (unfitted).
///
/// Call [`Fit::fit`] to run the algorithm and obtain a
/// [`FittedAgglomerativeClustering`].
///
/// # Type Parameters
///
/// - `F`: floating-point scalar type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct AgglomerativeClustering<F> {
    /// Target number of clusters.
    pub n_clusters: usize,
    /// Linkage strategy for computing inter-cluster distances.
    pub linkage: Linkage,
    /// Phantom to retain the float type parameter.
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float> AgglomerativeClustering<F> {
    /// Create a new `AgglomerativeClustering` with the given number of clusters.
    ///
    /// Uses default `linkage = Ward`.
    #[must_use]
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            linkage: Linkage::Ward,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the linkage criterion.
    #[must_use]
    pub fn with_linkage(mut self, linkage: Linkage) -> Self {
        self.linkage = linkage;
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fitted model
// ─────────────────────────────────────────────────────────────────────────────

/// Fitted Agglomerative Clustering model.
///
/// Stores per-sample cluster labels, the actual cluster count, and the
/// merge history (dendrogram).
///
/// There is intentionally **no** `predict` method: new data cannot be
/// assigned to clusters without re-running the full algorithm.
#[derive(Debug, Clone)]
pub struct FittedAgglomerativeClustering<F> {
    /// Cluster label for each training sample, shape `(n_samples,)`.
    /// Labels are in the range `0 .. n_clusters_`.
    pub labels_: Array1<usize>,
    /// The actual number of clusters formed.
    pub n_clusters_: usize,
    /// Merge history: each element `(i, j)` records that the clusters
    /// with internal IDs `i` and `j` were merged.  Length =
    /// `n_samples - n_clusters`.
    pub children_: Vec<(usize, usize)>,
    /// Phantom to retain the float type parameter.
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float> FittedAgglomerativeClustering<F> {
    /// Return the cluster label for each training sample.
    #[must_use]
    pub fn labels(&self) -> &Array1<usize> {
        &self.labels_
    }

    /// Return the number of clusters formed.
    #[must_use]
    pub fn n_clusters(&self) -> usize {
        self.n_clusters_
    }

    /// Return the merge tree: pairs of cluster IDs that were merged.
    ///
    /// The entries are in merge order (earliest merge first).
    #[must_use]
    pub fn children(&self) -> &[(usize, usize)] {
        &self.children_
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the squared Euclidean distance between two row slices.
#[inline]
fn sq_euclidean<F: Float>(a: &[F], b: &[F]) -> F {
    a.iter()
        .zip(b.iter())
        .fold(F::zero(), |acc, (&ai, &bi)| acc + (ai - bi) * (ai - bi))
}

/// Compute the full `n × n` pairwise squared-distance matrix.
fn pairwise_sq_dists<F: Float>(x: &Array2<F>) -> Vec<F> {
    let n = x.nrows();
    let mut d = vec![F::zero(); n * n];
    for i in 0..n {
        let ri = x.row(i);
        let si = ri.as_slice().unwrap_or(&[]);
        for j in (i + 1)..n {
            let rj = x.row(j);
            let sj = rj.as_slice().unwrap_or(&[]);
            let dist = sq_euclidean(si, sj);
            d[i * n + j] = dist;
            d[j * n + i] = dist;
        }
    }
    d
}

/// Find the (i, j) pair with the smallest value in `dist_mat` among the
/// currently active clusters.
fn find_min_pair(dist_mat: &[f64], active: &[usize]) -> (usize, usize) {
    let mut best_i = active[0];
    let mut best_j = active[1];
    let n = (dist_mat.len() as f64).sqrt() as usize;
    let mut best_val = f64::INFINITY;

    for (ai, &i) in active.iter().enumerate() {
        for &j in active.iter().skip(ai + 1) {
            let v = dist_mat[i * n + j];
            if v < best_val {
                best_val = v;
                best_i = i;
                best_j = j;
            }
        }
    }
    (best_i, best_j)
}

/// Return type of the internal `agglomerate` helper.
type AgglomerateResult = Result<(Array1<usize>, Vec<(usize, usize)>), FerroError>;

/// Generic helper: run agglomerative clustering returning `(labels, children)`.
///
/// We work entirely with `f64` internally and accept the input as a trait
/// object of `Float` by converting upfront.
fn agglomerate<F: Float>(
    x: &Array2<F>,
    n_clusters_target: usize,
    linkage: Linkage,
) -> AgglomerateResult {
    let n_samples = x.nrows();

    // Convert data to f64 for internal computation.
    let x_f64: Array2<f64> = x.mapv(|v| v.to_f64().unwrap_or(0.0));

    // Build pairwise squared-distance matrix (n × n, flat, row-major).
    let mut sq_dists = pairwise_sq_dists(&x_f64);
    let n = n_samples;

    // For Ward linkage we also need cluster sizes and sum-of-squares.
    // For others we just track sizes to apply Lance–Williams updates.
    let mut sizes: Vec<f64> = vec![1.0; n];

    // active[i] = current internal cluster ID of the i-th active position.
    let mut active: Vec<usize> = (0..n).collect();

    let mut children: Vec<(usize, usize)> = Vec::with_capacity(n - n_clusters_target);

    // cluster_id[i] = which leaf cluster i belongs to at the current merge step.
    // Initially each sample is its own cluster.
    let mut assignment: Vec<usize> = (0..n).collect();

    // Counter for new cluster IDs after merges (reuse the merged-into slot).
    // We track the merge history as pairs of original-or-merged IDs.

    while active.len() > n_clusters_target {
        // ── Find the two closest active clusters ────────────────────────────
        let (ci, cj) = find_min_pair(&sq_dists, &active);

        // Remove cj from active; ci absorbs cj.
        active.retain(|&id| id != cj);
        children.push((ci, cj));

        let ni = sizes[ci];
        let nj = sizes[cj];
        let new_size = ni + nj;

        // ── Update the distance matrix using Lance–Williams recurrence ───────
        // For the merged cluster (stored in slot ci), update dist to all
        // remaining active clusters.
        for &ck in &active {
            if ck == ci {
                continue;
            }
            let nk = sizes[ck];
            let d_ik = sq_dists[ci * n + ck];
            let d_jk = sq_dists[cj * n + ck];

            let new_dist = match linkage {
                Linkage::Single => {
                    if d_ik < d_jk {
                        d_ik
                    } else {
                        d_jk
                    }
                }
                Linkage::Complete => {
                    if d_ik > d_jk {
                        d_ik
                    } else {
                        d_jk
                    }
                }
                Linkage::Average => (ni * d_ik + nj * d_jk) / (ni + nj),
                Linkage::Ward => {
                    // Ward: squared Euclidean distance between new centroid
                    // and existing centroid, weighted by sizes.
                    // Lance–Williams for Ward:
                    // d(ij, k) = ((n_i + n_k)/(n_i+n_j+n_k)) * d(i,k)
                    //          + ((n_j + n_k)/(n_i+n_j+n_k)) * d(j,k)
                    //          - (n_k      /(n_i+n_j+n_k)) * d(i,j)
                    let d_ij = sq_dists[ci * n + cj];
                    let denom = ni + nj + nk;
                    ((ni + nk) / denom) * d_ik + ((nj + nk) / denom) * d_jk - (nk / denom) * d_ij
                }
            };

            sq_dists[ci * n + ck] = new_dist;
            sq_dists[ck * n + ci] = new_dist;
        }

        sizes[ci] = new_size;

        // Redirect all samples assigned to cj → ci.
        for s in &mut assignment {
            if *s == cj {
                *s = ci;
            }
        }
    }

    // ── Re-label active cluster IDs as 0 .. n_clusters_target ───────────────
    let mut id_map: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for (new_id, &cluster_id) in active.iter().enumerate() {
        id_map.insert(cluster_id, new_id);
    }
    let labels: Array1<usize> = assignment
        .iter()
        .map(|id| *id_map.get(id).unwrap_or(&0))
        .collect();

    Ok((labels, children))
}

// ─────────────────────────────────────────────────────────────────────────────
// Trait impl: Fit
// ─────────────────────────────────────────────────────────────────────────────

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for AgglomerativeClustering<F> {
    type Fitted = FittedAgglomerativeClustering<F>;
    type Error = FerroError;

    /// Run agglomerative clustering on `x`.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `n_clusters == 0`.
    /// - [`FerroError::InsufficientSamples`] if `n_samples < n_clusters`.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedAgglomerativeClustering<F>, FerroError> {
        if self.n_clusters == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_clusters".into(),
                reason: "must be at least 1".into(),
            });
        }

        let n_samples = x.nrows();

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: self.n_clusters,
                actual: 0,
                context: "AgglomerativeClustering requires at least n_clusters samples".into(),
            });
        }

        if n_samples < self.n_clusters {
            return Err(FerroError::InsufficientSamples {
                required: self.n_clusters,
                actual: n_samples,
                context: "AgglomerativeClustering requires at least n_clusters samples".into(),
            });
        }

        let (labels, children) = agglomerate(x, self.n_clusters, self.linkage)?;

        Ok(FittedAgglomerativeClustering {
            labels_: labels,
            n_clusters_: self.n_clusters,
            children_: children,
            _marker: std::marker::PhantomData,
        })
    }
}

impl<F: Float + Send + Sync + 'static> AgglomerativeClustering<F> {
    /// Fit on `x` and return the cluster labels for those samples in one
    /// call. Equivalent to sklearn `ClusterMixin.fit_predict`.
    ///
    /// # Errors
    ///
    /// Forwards any error from [`Fit::fit`].
    pub fn fit_predict(&self, x: &Array2<F>) -> Result<Array1<usize>, FerroError> {
        let fitted = self.fit(x, &())?;
        Ok(fitted.labels().clone())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Two well-separated blobs.
    fn make_two_blobs() -> Array2<f64> {
        Array2::from_shape_vec(
            (8, 2),
            vec![
                0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.05, 0.05, 10.0, 10.0, 10.1, 10.0, 10.0, 10.1,
                10.05, 10.05,
            ],
        )
        .unwrap()
    }

    /// Three well-separated blobs.
    fn make_three_blobs() -> Array2<f64> {
        Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.0, 0.1, 0.1, -0.1, 0.1, 10.0, 10.0, 10.1, 10.1, 9.9, 10.1, 0.0, 10.0, 0.1,
                10.1, -0.1, 9.9,
            ],
        )
        .unwrap()
    }

    // ── Construction ────────────────────────────────────────────────────────

    #[test]
    fn test_new_defaults() {
        let model = AgglomerativeClustering::<f64>::new(3);
        assert_eq!(model.n_clusters, 3);
        assert_eq!(model.linkage, Linkage::Ward);
    }

    #[test]
    fn test_with_linkage() {
        let model = AgglomerativeClustering::<f64>::new(2).with_linkage(Linkage::Complete);
        assert_eq!(model.linkage, Linkage::Complete);
    }

    // ── Error conditions ────────────────────────────────────────────────────

    #[test]
    fn test_zero_clusters_error() {
        let x = make_two_blobs();
        let result = AgglomerativeClustering::<f64>::new(0).fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_data_error() {
        let x = Array2::<f64>::zeros((0, 2));
        let result = AgglomerativeClustering::<f64>::new(2).fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_more_clusters_than_samples_error() {
        let x = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 2.0, 2.0]).unwrap();
        let result = AgglomerativeClustering::<f64>::new(5).fit(&x, &());
        assert!(result.is_err());
    }

    // ── Ward linkage ────────────────────────────────────────────────────────

    #[test]
    fn test_ward_two_blobs() {
        let x = make_two_blobs();
        let fitted = AgglomerativeClustering::<f64>::new(2)
            .with_linkage(Linkage::Ward)
            .fit(&x, &())
            .unwrap();
        let labels = fitted.labels();
        // First 4 should be in the same cluster; last 4 in another.
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[0], labels[3]);
        assert_eq!(labels[4], labels[5]);
        assert_eq!(labels[4], labels[6]);
        assert_eq!(labels[4], labels[7]);
        assert_ne!(labels[0], labels[4]);
    }

    #[test]
    fn test_ward_three_blobs() {
        let x = make_three_blobs();
        let fitted = AgglomerativeClustering::<f64>::new(3)
            .with_linkage(Linkage::Ward)
            .fit(&x, &())
            .unwrap();
        let labels = fitted.labels();
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[6], labels[7]);
        assert_ne!(labels[0], labels[3]);
        assert_ne!(labels[0], labels[6]);
        assert_ne!(labels[3], labels[6]);
    }

    // ── Complete linkage ────────────────────────────────────────────────────

    #[test]
    fn test_complete_two_blobs() {
        let x = make_two_blobs();
        let fitted = AgglomerativeClustering::<f64>::new(2)
            .with_linkage(Linkage::Complete)
            .fit(&x, &())
            .unwrap();
        let labels = fitted.labels();
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[4]);
    }

    #[test]
    fn test_complete_three_blobs() {
        let x = make_three_blobs();
        let fitted = AgglomerativeClustering::<f64>::new(3)
            .with_linkage(Linkage::Complete)
            .fit(&x, &())
            .unwrap();
        let labels = fitted.labels();
        assert_ne!(labels[0], labels[3]);
        assert_ne!(labels[0], labels[6]);
    }

    // ── Average linkage ─────────────────────────────────────────────────────

    #[test]
    fn test_average_two_blobs() {
        let x = make_two_blobs();
        let fitted = AgglomerativeClustering::<f64>::new(2)
            .with_linkage(Linkage::Average)
            .fit(&x, &())
            .unwrap();
        let labels = fitted.labels();
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[4]);
    }

    #[test]
    fn test_average_three_blobs() {
        let x = make_three_blobs();
        let fitted = AgglomerativeClustering::<f64>::new(3)
            .with_linkage(Linkage::Average)
            .fit(&x, &())
            .unwrap();
        let labels = fitted.labels();
        assert_ne!(labels[0], labels[3]);
        assert_ne!(labels[0], labels[6]);
    }

    // ── Single linkage ──────────────────────────────────────────────────────

    #[test]
    fn test_single_two_blobs() {
        let x = make_two_blobs();
        let fitted = AgglomerativeClustering::<f64>::new(2)
            .with_linkage(Linkage::Single)
            .fit(&x, &())
            .unwrap();
        let labels = fitted.labels();
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[4]);
    }

    #[test]
    fn test_single_three_blobs() {
        let x = make_three_blobs();
        let fitted = AgglomerativeClustering::<f64>::new(3)
            .with_linkage(Linkage::Single)
            .fit(&x, &())
            .unwrap();
        let labels = fitted.labels();
        assert_ne!(labels[0], labels[3]);
        assert_ne!(labels[0], labels[6]);
    }

    // ── Label properties ─────────────────────────────────────────────────────

    #[test]
    fn test_label_count_equals_n_samples() {
        let x = make_two_blobs();
        let fitted = AgglomerativeClustering::<f64>::new(2).fit(&x, &()).unwrap();
        assert_eq!(fitted.labels().len(), x.nrows());
    }

    #[test]
    fn test_labels_in_valid_range() {
        let x = make_three_blobs();
        let fitted = AgglomerativeClustering::<f64>::new(3).fit(&x, &()).unwrap();
        for &l in fitted.labels() {
            assert!(l < 3, "label {l} out of range");
        }
    }

    #[test]
    fn test_n_clusters_matches_config() {
        let x = make_three_blobs();
        let fitted = AgglomerativeClustering::<f64>::new(3).fit(&x, &()).unwrap();
        assert_eq!(fitted.n_clusters(), 3);
    }

    // ── Children (merge tree) ────────────────────────────────────────────────

    #[test]
    fn test_children_length() {
        let x = make_two_blobs(); // 8 samples, 2 clusters → 6 merges
        let fitted = AgglomerativeClustering::<f64>::new(2).fit(&x, &()).unwrap();
        assert_eq!(fitted.children().len(), x.nrows() - 2);
    }

    #[test]
    fn test_children_empty_when_n_clusters_equals_n_samples() {
        let x = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 5.0, 5.0, 10.0, 10.0]).unwrap();
        let fitted = AgglomerativeClustering::<f64>::new(3).fit(&x, &()).unwrap();
        assert!(fitted.children().is_empty());
    }

    // ── Special cases ─────────────────────────────────────────────────────────

    #[test]
    fn test_single_cluster() {
        let x = make_two_blobs();
        let fitted = AgglomerativeClustering::<f64>::new(1).fit(&x, &()).unwrap();
        // All samples should be in cluster 0.
        for &l in fitted.labels() {
            assert_eq!(l, 0);
        }
    }

    #[test]
    fn test_n_clusters_equals_n_samples() {
        let x = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 5.0, 5.0, 10.0, 10.0]).unwrap();
        let fitted = AgglomerativeClustering::<f64>::new(3).fit(&x, &()).unwrap();
        // Each sample is its own cluster; labels should all be distinct.
        let labels = fitted.labels();
        assert_ne!(labels[0], labels[1]);
        assert_ne!(labels[0], labels[2]);
        assert_ne!(labels[1], labels[2]);
    }

    #[test]
    fn test_single_sample_single_cluster() {
        let x = Array2::from_shape_vec((1, 2), vec![3.0, 4.0]).unwrap();
        let fitted = AgglomerativeClustering::<f64>::new(1).fit(&x, &()).unwrap();
        assert_eq!(fitted.labels()[0], 0);
        assert_eq!(fitted.n_clusters(), 1);
        assert!(fitted.children().is_empty());
    }

    #[test]
    fn test_1d_data() {
        let x = Array2::from_shape_vec((6, 1), vec![0.0, 0.1, -0.1, 100.0, 100.1, 99.9]).unwrap();
        let fitted = AgglomerativeClustering::<f64>::new(2).fit(&x, &()).unwrap();
        let labels = fitted.labels();
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn test_f32_support() {
        let x = Array2::<f32>::from_shape_vec(
            (6, 2),
            vec![
                0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 10.0, 10.0, 10.1, 10.0, 10.0, 10.1,
            ],
        )
        .unwrap();
        let fitted = AgglomerativeClustering::<f32>::new(2).fit(&x, &()).unwrap();
        assert_eq!(fitted.labels().len(), 6);
        let labels = fitted.labels();
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[3], labels[4]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn test_identical_points() {
        // All points identical → all should be in the same cluster.
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).unwrap();
        let fitted = AgglomerativeClustering::<f64>::new(1).fit(&x, &()).unwrap();
        for &l in fitted.labels() {
            assert_eq!(l, 0);
        }
    }
}
