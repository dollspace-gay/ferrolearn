//! BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies).
//!
//! This module provides [`Birch`], an incremental clustering algorithm designed
//! for large datasets. It builds a **CF (Clustering Feature) tree** to
//! summarize the data into subclusters, then optionally applies a final
//! clustering step (e.g., K-Means) to produce the desired number of clusters.
//!
//! # Algorithm
//!
//! 1. Build a CF tree where each leaf holds a subcluster defined by its count
//!    `N`, linear sum `LS`, and squared sum `SS`.
//! 2. A new point is inserted into the closest leaf subcluster. If the
//!    subcluster radius exceeds `threshold`, the leaf is split.
//! 3. If `n_clusters` is set, run Agglomerative Clustering (Ward linkage)
//!    on the subcluster centroids to produce the final labels.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_cluster::Birch;
//! use ferrolearn_core::Fit;
//! use ndarray::Array2;
//!
//! let x = Array2::from_shape_vec((6, 2), vec![
//!     0.0, 0.0,  0.1, 0.0,  0.0, 0.1,
//!     8.0, 8.0,  8.1, 8.0,  8.0, 8.1,
//! ]).unwrap();
//!
//! let model = Birch::<f64>::new()
//!     .with_threshold(0.5)
//!     .with_n_clusters(2);
//! let fitted = model.fit(&x, &()).unwrap();
//! assert_eq!(fitted.labels().len(), 6);
//! ```
//!
//! # `## REQ status`
//!
//! Binary classification (R-DEFER-2): two states only — SHIPPED needs impl + a
//! non-test production consumer + green verification + symbol-anchor + sklearn
//! `file:line`; NOT-STARTED carries the open prereq blocker. `Birch` /
//! `FittedBirch` are existing pub APIs re-exported at the crate root (`pub use
//! birch::{Birch, FittedBirch}` in `ferrolearn-cluster/src/lib.rs`) with a real
//! PyO3 binding — `#[pyclass(name = "_RsBirch")] struct RsBirch` in
//! `ferrolearn-python/src/extras.rs` (its `fn fit` calls
//! `ferrolearn_cluster::Birch::<f64>::new().with_threshold(...).with_n_clusters(...)`
//! then `.fit(...)`, and `#[getter] labels_` surfaces `f.labels()`), registered
//! via `m.add_class::<extras::RsBirch>()` in `ferrolearn-python/src/lib.rs`,
//! surfaced as `ferrolearn.Birch` (`fit` / `labels_`). So
//! `import ferrolearn; ferrolearn.Birch(...).fit(X).labels_` is the non-test
//! production consumer of the core fit/labels contract (the binding exposes ONLY
//! `n_clusters`/`threshold`/`fit`/`labels_` — not `subcluster_centers_`,
//! `predict`, or `transform`).
//!
//! **Honest assessment (R-HONEST-3).** The mean-based `subcluster_centers_` (CF
//! centroid `LS/N`, REQ-1) and the `labels_` PARTITION via the global
//! Agglomerative-Ward step (REQ-3) genuinely VALUE-match the live sklearn 1.5.2
//! oracle — but ONLY in the regime where the number of leaf subclusters stays
//! `<= branching_factor`. The CF data structure ferrolearn implements is a FLAT
//! `Vec<ClusteringFeature>` (`fn build_cf_tree`), NOT a balanced CF-tree: once the
//! subcluster count would exceed `branching_factor` it MERGES the two closest
//! subclusters (`fn find_closest_pair` + `fn merge`) instead of splitting a leaf
//! node and growing the tree, so the leaf-subcluster count is capped at
//! `branching_factor`. sklearn's `_split_node` (`_birch.py:48-108`) splits a full
//! leaf into two and grows the tree, so its leaf-subcluster count is unbounded —
//! the root structural divergence (REQ-2, #954): on `spread60`
//! (`RandomState(1).rand(60,2)*20`) at `branching_factor=5`, sklearn yields 37
//! leaf subclusters, ferrolearn caps at 5. Scope caveats on the SHIPPED rows:
//! REQ-1 matches as a SET of centroid rows (sklearn leaf-order vs ferrolearn
//! insertion-order differ; the VALUES match, not row order) and REQ-3 matches the
//! partition UP TO A LABEL PERMUTATION (Agglomerative-Ward label ordering), NOT
//! absolute labels. The whole `predict`/`transform`/`partial_fit`/
//! `subcluster_labels_` surface and the `compute_labels`/`copy` params are absent.
//! Cites use symbol anchors (ferrolearn) / `file:line` (sklearn 1.5.2, commit
//! 156ef14). Live oracle = installed sklearn 1.5.2, run from `/tmp`. (REQ
//! numbering follows `.design/cluster/birch.md`.)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`subcluster_centers_` VALUE, in `n_subclusters <= branching_factor` regime) | SHIPPED | impl `fn centroid in birch.rs` (CF `ls[i]/n`) → `Fit::fit` builds `subcluster_centers_`; value-matches sklearn `_CFSubcluster.update` centroid (`_birch.py:319`) read off leaves (`_birch.py:590-591`) to full f64 **as a SET of centroid rows** (sklearn leaf-order vs ferrolearn insertion-order differ; VALUES match, not row order). Non-test consumer: `RsBirch` (`ferrolearn-python/src/extras.rs`, via `Birch::fit`) + crate re-export. Green guards (`tests/divergence_birch.rs`): `green_birch_subcluster_centers_blobs8` (`blobs8` `[[0.0375,…],[10.0375,…]]`), `green_birch_subcluster_centers_blobs20` (all 10 centroids ~1e-9). **Caveat (REQ-2/#954): contract holds only when sklearn yields `<= branching_factor` subclusters** — diverges otherwise (`spread60`: sklearn 37 vs ferrolearn 5). |
//! | REQ-2 (CF-tree leaf splitting) | NOT-STARTED | open prereq blocker **#954**. sklearn `_CFNode.insert_cf_subcluster` (`_birch.py:196-263`) recurses into child nodes and `_split_node` (`_birch.py:48-108`) splits a full leaf into two leaves, so leaf-subcluster count is unbounded by `branching_factor`. ferrolearn `fn build_cf_tree` keeps a FLAT `Vec<ClusteringFeature>` capped at `branching_factor`, MERGING the two closest CFs on overflow (`fn find_closest_pair` + `fn merge`). Live oracle (`spread60` = `RandomState(1).rand(60,2)*20`, `branching_factor=5`): sklearn 37 subclusters vs ferrolearn 5. **Root structural divergence — a CF-tree rewrite, not a minimal fix.** |
//! | REQ-3 (`labels_` partition via global Agglo-Ward, up to a label permutation, in-regime) | SHIPPED | impl `Fit::fit in birch.rs` runs `AgglomerativeClustering::new(k).with_linkage(Linkage::Ward)` on `subcluster_centers_`, mirroring `_global_clustering` int branch (`_birch.py:703-738`, default Agglomerative = Ward). PARTITION value-matches sklearn **up to a label permutation** on few-subcluster fixtures (absolute label values differ — Agglomerative label ordering). Non-test consumer: `RsBirch::labels_` (`ferrolearn-python/src/extras.rs`) + crate re-export. Green guard: `green_birch_labels_partition_blobs8` (`blobs8` co-membership; sklearn `[1,1,1,1,0,0,0,0]` vs ferrolearn label-swapped, same partition). **Caveat**: gated on REQ-2 (#954) in the `> branching_factor` regime and on REQ-5 (#956) for the out-of-sample `_predict` path. |
//! | REQ-4 (`n_clusters=3` default + estimator form + error ABI) | NOT-STARTED | open prereq blocker **#955**. sklearn `n_clusters=3` default (`_birch.py:496`), accepts `None`/`int>=1`/a `sklearn.cluster` estimator (`_birch.py:486`, `:731-735`), raises `InvalidParameterError`. ferrolearn `fn new()` defaults `None`, `Option<usize>` (no estimator-instance form), errors `FerroError::InvalidParameter`. `Birch().n_clusters` = 3 (sklearn) vs `None` (ferrolearn + binding). |
//! | REQ-5 (`predict` + `transform` out-of-sample) | NOT-STARTED | open prereq blocker **#956**. sklearn `predict`/`_predict` = `pairwise_distances_argmin(X, subcluster_centers_)` → `subcluster_labels_[argmin]` (`_birch.py:651-679`); `transform` = `euclidean_distances(X, subcluster_centers_)` (`_birch.py:681-701`). `FittedBirch<F>` has neither (`transform(docstring6)[0]=[0,2]`, `predict=[0,0,0,1,1,1]`). Also: ferrolearn `labels_` threads points through their insertion CF, NOT argmin over final centroids. |
//! | REQ-6 (`subcluster_labels_` attribute) | NOT-STARTED | open prereq blocker **#957**. sklearn `subcluster_labels_` = global label per subcluster (`_birch.py:723`/`:735`), the bridge `predict` uses. `FittedBirch` exposes `labels()`/`subcluster_centers()`/`n_clusters()` but no `subcluster_labels_` accessor; `RsBirch` cannot surface it. |
//! | REQ-7 (`not_enough_centroids` skip + ConvergenceWarning) | NOT-STARTED | open prereq blocker **#958**. sklearn, when `len(centroids) < n_clusters`, SKIPS the global step, sets `subcluster_labels_ = arange(len)`, warns `ConvergenceWarning` (`_birch.py:716`/`:722-730`). ferrolearn `Fit::fit` CLAMPS `k.min(n_subclusters)` and runs Agglomerative on the clamped k — no warning, `n_clusters_` silently clamped. |
//! | REQ-8 (`partial_fit` online learning) | NOT-STARTED | open prereq blocker **#959**. sklearn `partial_fit` (`_birch.py:613-638`) incrementally inserts without rebuilding (`X=None` → global step only). ferrolearn offers only batch `Fit::fit`; no incremental API. |
//! | REQ-9 (`compute_labels` + `copy` params) | NOT-STARTED | open prereq blocker **#960**. sklearn `compute_labels=True` (`_birch.py:497`, gates `labels_` `:709`/`:737`) + `copy=True` (`_birch.py:498`, gates in-place `:534`). `Birch<F>` (`threshold`/`branching_factor`/`n_clusters`) has neither. |
//! | REQ-10 (threshold criterion form) | SHIPPED | impl `fn would_exceed_threshold in birch.rs` computes `variance = new_ss/n - centroid_sq_norm`, `radius = sqrt(max(0,variance))`, tests `radius > threshold` — algebraically EQUIVALENT to sklearn `sq_radius <= threshold**2` (`_CFSubcluster.merge_subcluster`, `_birch.py:340-342`; `r>t <=> r^2>t^2` for non-negative `r,t`), including the negative-variance clamp (sklearn `sqrt(max(0,...))`). Consumer: `fn build_cf_tree` (in-crate) → `Fit::fit` → `RsBirch`. Green guard: `green_birch_threshold_validation` covers the `threshold>0` / `branching_factor>=2` bounds (sklearn `InvalidParameterError`). |
//! | REQ-11 (ferray substrate) | NOT-STARTED | open prereq blocker **#961**. `birch.rs` imports `ndarray::{Array1, Array2}` + `num_traits::Float`; not migrated to `ferray-core` / `ferray::linalg` (R-SUBSTRATE-1/2). |

use crate::agglomerative::{AgglomerativeClustering, Linkage};
use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::Fit;
use ndarray::{Array1, Array2};
use num_traits::Float;

/// BIRCH clustering configuration (unfitted).
///
/// Holds hyperparameters for the BIRCH algorithm. Call [`Fit::fit`]
/// to run the algorithm and produce a [`FittedBirch`].
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct Birch<F> {
    /// The maximum radius of a subcluster in the CF tree.
    pub threshold: F,
    /// Maximum number of children (subclusters) per CF tree node.
    pub branching_factor: usize,
    /// Optional number of clusters for the final clustering step.
    /// If `None`, the subclusters themselves are the final clusters.
    pub n_clusters: Option<usize>,
}

impl<F: Float> Birch<F> {
    /// Create a new `Birch` with default parameters.
    ///
    /// Defaults: `threshold = 0.5`, `branching_factor = 50`, `n_clusters = None`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            threshold: F::from(0.5).unwrap_or_else(F::epsilon),
            branching_factor: 50,
            n_clusters: None,
        }
    }

    /// Set the subcluster radius threshold.
    #[must_use]
    pub fn with_threshold(mut self, threshold: F) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set the maximum branching factor.
    #[must_use]
    pub fn with_branching_factor(mut self, branching_factor: usize) -> Self {
        self.branching_factor = branching_factor;
        self
    }

    /// Set the number of clusters for the final step.
    #[must_use]
    pub fn with_n_clusters(mut self, n_clusters: usize) -> Self {
        self.n_clusters = Some(n_clusters);
        self
    }
}

impl<F: Float> Default for Birch<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted BIRCH model.
///
/// Stores the cluster labels and subcluster centers from the training run.
///
/// BIRCH does **not** implement [`Predict`](ferrolearn_core::Predict).
#[derive(Debug, Clone)]
pub struct FittedBirch<F> {
    /// Cluster label for each training sample.
    labels_: Array1<usize>,
    /// Subcluster center coordinates, shape `(n_subclusters, n_features)`.
    subcluster_centers_: Array2<F>,
    /// Number of clusters formed.
    n_clusters_: usize,
}

impl<F: Float> FittedBirch<F> {
    /// Return the cluster labels for the training data.
    #[must_use]
    pub fn labels(&self) -> &Array1<usize> {
        &self.labels_
    }

    /// Return the subcluster center coordinates.
    #[must_use]
    pub fn subcluster_centers(&self) -> &Array2<F> {
        &self.subcluster_centers_
    }

    /// Return the number of clusters formed.
    #[must_use]
    pub fn n_clusters(&self) -> usize {
        self.n_clusters_
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal: Clustering Feature (CF) structures
// ─────────────────────────────────────────────────────────────────────────────

/// A Clustering Feature: (N, LS, SS).
#[derive(Debug, Clone)]
struct ClusteringFeature<F> {
    /// Number of data points in this subcluster.
    n: usize,
    /// Linear sum of data points, shape `(n_features,)`.
    ls: Vec<F>,
    /// Sum of squared norms of data points.
    ss: F,
    /// Indices of original data points in this subcluster.
    point_indices: Vec<usize>,
}

impl<F: Float> ClusteringFeature<F> {
    /// Create a new CF from a single data point.
    fn from_point(point: &[F], index: usize) -> Self {
        let ss = point.iter().fold(F::zero(), |acc, &v| acc + v * v);
        Self {
            n: 1,
            ls: point.to_vec(),
            ss,
            point_indices: vec![index],
        }
    }

    /// Compute the centroid of this subcluster.
    fn centroid(&self) -> Vec<F> {
        let n = F::from(self.n).unwrap_or_else(F::one);
        self.ls.iter().map(|&v| v / n).collect()
    }

    /// Check if adding a point would exceed the threshold.
    fn would_exceed_threshold(&self, point: &[F], threshold: F) -> bool {
        // Compute the radius if we were to add this point.
        let new_n = self.n + 1;
        let n_f = F::from(new_n).unwrap_or_else(F::one);
        let new_ss = self.ss + point.iter().fold(F::zero(), |acc, &v| acc + v * v);
        let new_ls: Vec<F> = self
            .ls
            .iter()
            .zip(point.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        let centroid_sq_norm = new_ls
            .iter()
            .fold(F::zero(), |acc, &v| acc + (v / n_f) * (v / n_f));
        let variance = new_ss / n_f - centroid_sq_norm;
        let radius = if variance > F::zero() {
            variance.sqrt()
        } else {
            F::zero()
        };
        radius > threshold
    }

    /// Add a point to this subcluster.
    fn absorb_point(&mut self, point: &[F], index: usize) {
        self.n += 1;
        for (ls, &p) in self.ls.iter_mut().zip(point.iter()) {
            *ls = *ls + p;
        }
        self.ss = self.ss + point.iter().fold(F::zero(), |acc, &v| acc + v * v);
        self.point_indices.push(index);
    }

    /// Merge another CF into this one.
    fn merge(&mut self, other: &ClusteringFeature<F>) {
        self.n += other.n;
        for (ls, &ols) in self.ls.iter_mut().zip(other.ls.iter()) {
            *ls = *ls + ols;
        }
        self.ss = self.ss + other.ss;
        self.point_indices.extend_from_slice(&other.point_indices);
    }

    /// Compute the distance from this CF's centroid to a point.
    fn distance_to_point(&self, point: &[F]) -> F {
        let centroid = self.centroid();
        centroid
            .iter()
            .zip(point.iter())
            .fold(F::zero(), |acc, (&c, &p)| acc + (c - p) * (c - p))
            .sqrt()
    }
}

/// Build the CF tree and return the subclusters.
fn build_cf_tree<F: Float>(
    x: &Array2<F>,
    threshold: F,
    branching_factor: usize,
) -> Vec<ClusteringFeature<F>> {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    if n_samples == 0 {
        return Vec::new();
    }

    let mut subclusters: Vec<ClusteringFeature<F>> = Vec::new();

    for i in 0..n_samples {
        let row = x.row(i);
        let point: Vec<F> = row.to_vec();

        if subclusters.is_empty() {
            subclusters.push(ClusteringFeature::from_point(&point, i));
            continue;
        }

        // Find the closest subcluster.
        let mut best_idx = 0;
        let mut best_dist = F::max_value();
        for (j, sc) in subclusters.iter().enumerate() {
            let d = sc.distance_to_point(&point);
            if d < best_dist {
                best_dist = d;
                best_idx = j;
            }
        }

        // Try to absorb the point into the closest subcluster.
        if !subclusters[best_idx].would_exceed_threshold(&point, threshold) {
            subclusters[best_idx].absorb_point(&point, i);
        } else if subclusters.len() < branching_factor {
            // Create a new subcluster.
            subclusters.push(ClusteringFeature::from_point(&point, i));
        } else {
            // Need to split: find the two closest subclusters and merge them,
            // then add the new point as a new subcluster.
            if subclusters.len() >= 2 {
                let (merge_i, merge_j) = find_closest_pair(&subclusters);
                let merged = {
                    let mut m = subclusters[merge_i].clone();
                    m.merge(&subclusters[merge_j]);
                    m
                };
                // Remove the two and add the merged one.
                let max_idx = merge_i.max(merge_j);
                let min_idx = merge_i.min(merge_j);
                subclusters.remove(max_idx);
                subclusters.remove(min_idx);
                subclusters.push(merged);
            }
            subclusters.push(ClusteringFeature::from_point(&point, i));
        }
    }

    // Ensure we have at least one feature dimension initialized.
    let _ = n_features;

    subclusters
}

/// Find the two closest subclusters by centroid distance.
fn find_closest_pair<F: Float>(subclusters: &[ClusteringFeature<F>]) -> (usize, usize) {
    let mut best_i = 0;
    let mut best_j = 1;
    let mut best_dist = F::max_value();

    let centroids: Vec<Vec<F>> = subclusters
        .iter()
        .map(ClusteringFeature::centroid)
        .collect();

    for i in 0..centroids.len() {
        for j in (i + 1)..centroids.len() {
            let d: F = centroids[i]
                .iter()
                .zip(centroids[j].iter())
                .fold(F::zero(), |acc, (&a, &b)| acc + (a - b) * (a - b));
            if d < best_dist {
                best_dist = d;
                best_i = i;
                best_j = j;
            }
        }
    }

    (best_i, best_j)
}

// ─────────────────────────────────────────────────────────────────────────────
// Trait impl: Fit
// ─────────────────────────────────────────────────────────────────────────────

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for Birch<F> {
    type Fitted = FittedBirch<F>;
    type Error = FerroError;

    /// Fit the BIRCH model to the data.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if `threshold` is not positive
    /// or `branching_factor` is less than 2.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedBirch<F>, FerroError> {
        if self.threshold <= F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "threshold".into(),
                reason: "must be positive".into(),
            });
        }

        if self.branching_factor < 2 {
            return Err(FerroError::InvalidParameter {
                name: "branching_factor".into(),
                reason: "must be at least 2".into(),
            });
        }

        if let Some(k) = self.n_clusters
            && k == 0
        {
            return Err(FerroError::InvalidParameter {
                name: "n_clusters".into(),
                reason: "must be at least 1".into(),
            });
        }

        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples == 0 {
            return Ok(FittedBirch {
                labels_: Array1::zeros(0),
                subcluster_centers_: Array2::zeros((0, n_features)),
                n_clusters_: 0,
            });
        }

        // Build the CF tree.
        let subclusters = build_cf_tree(x, self.threshold, self.branching_factor);
        let n_subclusters = subclusters.len();

        // Compute subcluster centroids.
        let centroids: Vec<Vec<F>> = subclusters
            .iter()
            .map(ClusteringFeature::centroid)
            .collect();

        // Build the subcluster centers array.
        let mut centers_data = vec![F::zero(); n_subclusters * n_features];
        for (i, centroid) in centroids.iter().enumerate() {
            for (j, &val) in centroid.iter().enumerate() {
                centers_data[i * n_features + j] = val;
            }
        }
        let subcluster_centers = Array2::from_shape_vec((n_subclusters, n_features), centers_data)
            .map_err(|_| FerroError::NumericalInstability {
                message: "failed to construct subcluster centers matrix".into(),
            })?;

        // Determine final labels.
        let n_clusters;
        let mut labels = Array1::zeros(n_samples);

        if let Some(k) = self.n_clusters {
            // Run AgglomerativeClustering (Ward linkage) on the subcluster
            // centroids. This is what sklearn does and avoids the bad
            // convergence that naive-init KMeans produces when the CF-tree
            // inserts subclusters sequentially from the same spatial region.
            let k_actual = k.min(n_subclusters);
            let agglo = AgglomerativeClustering::<F>::new(k_actual).with_linkage(Linkage::Ward);
            let fitted_agglo = agglo.fit(&subcluster_centers, &()).map_err(|e| {
                FerroError::NumericalInstability {
                    message: format!(
                        "agglomerative clustering on subcluster centroids failed: {e}"
                    ),
                }
            })?;
            let subcluster_labels = fitted_agglo.labels();
            n_clusters = k_actual;

            // Map each point to the final label through its subcluster.
            for (sc_idx, sc) in subclusters.iter().enumerate() {
                let final_label = subcluster_labels[sc_idx];
                for &pt_idx in &sc.point_indices {
                    labels[pt_idx] = final_label;
                }
            }
        } else {
            // Each subcluster is a cluster.
            n_clusters = n_subclusters;
            for (sc_idx, sc) in subclusters.iter().enumerate() {
                for &pt_idx in &sc.point_indices {
                    labels[pt_idx] = sc_idx;
                }
            }
        }

        Ok(FittedBirch {
            labels_: labels,
            subcluster_centers_: subcluster_centers,
            n_clusters_: n_clusters,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Birch<F> {
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

    #[test]
    fn test_two_clusters_with_n_clusters() {
        let x = make_two_blobs();
        let model = Birch::<f64>::new().with_threshold(0.5).with_n_clusters(2);
        let fitted = model.fit(&x, &()).unwrap();

        let labels = fitted.labels();
        assert_eq!(labels.len(), 8);

        // First 4 should share a label, last 4 should share another.
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[0], labels[3]);
        assert_eq!(labels[4], labels[5]);
        assert_eq!(labels[4], labels[6]);
        assert_eq!(labels[4], labels[7]);
        assert_ne!(labels[0], labels[4]);
    }

    #[test]
    fn test_three_clusters() {
        let x = make_three_blobs();
        let model = Birch::<f64>::new().with_threshold(0.5).with_n_clusters(3);
        let fitted = model.fit(&x, &()).unwrap();

        let labels = fitted.labels();
        assert_eq!(labels.len(), 9);

        // Points within each blob should have the same label.
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[3], labels[5]);
        assert_eq!(labels[6], labels[7]);
        assert_eq!(labels[6], labels[8]);
    }

    #[test]
    fn test_subclusters_without_n_clusters() {
        let x = make_two_blobs();
        let model = Birch::<f64>::new().with_threshold(0.5);
        let fitted = model.fit(&x, &()).unwrap();

        assert_eq!(fitted.labels().len(), 8);
        assert!(
            fitted.n_clusters() >= 1,
            "should have at least 1 subcluster"
        );
        assert!(
            fitted.subcluster_centers().nrows() >= 1,
            "should have at least 1 subcluster center"
        );
    }

    #[test]
    fn test_threshold_effect_on_subclusters() {
        let x = make_two_blobs();

        // Small threshold = more subclusters.
        let model_small = Birch::<f64>::new().with_threshold(0.01);
        let fitted_small = model_small.fit(&x, &()).unwrap();

        // Large threshold = fewer subclusters.
        let model_large = Birch::<f64>::new().with_threshold(100.0);
        let fitted_large = model_large.fit(&x, &()).unwrap();

        assert!(
            fitted_small.subcluster_centers().nrows() >= fitted_large.subcluster_centers().nrows(),
            "smaller threshold should produce at least as many subclusters"
        );
    }

    #[test]
    fn test_subcluster_centers_shape() {
        let x = make_two_blobs();
        let model = Birch::<f64>::new().with_threshold(0.5);
        let fitted = model.fit(&x, &()).unwrap();

        let centers = fitted.subcluster_centers();
        assert_eq!(centers.ncols(), 2); // Same number of features.
        assert!(centers.nrows() >= 1);
    }

    #[test]
    fn test_empty_data() {
        let x = Array2::<f64>::zeros((0, 2));
        let model = Birch::<f64>::new();
        let fitted = model.fit(&x, &()).unwrap();

        assert_eq!(fitted.labels().len(), 0);
        assert_eq!(fitted.subcluster_centers().nrows(), 0);
        assert_eq!(fitted.n_clusters(), 0);
    }

    #[test]
    fn test_single_sample() {
        let x = Array2::from_shape_vec((1, 2), vec![5.0, 5.0]).unwrap();
        let model = Birch::<f64>::new().with_threshold(0.5);
        let fitted = model.fit(&x, &()).unwrap();

        assert_eq!(fitted.labels().len(), 1);
        assert_eq!(fitted.subcluster_centers().nrows(), 1);
    }

    #[test]
    fn test_invalid_threshold() {
        let x = make_two_blobs();
        let model = Birch::<f64>::new().with_threshold(-1.0);
        assert!(model.fit(&x, &()).is_err());
    }

    #[test]
    fn test_invalid_branching_factor() {
        let x = make_two_blobs();
        let model = Birch::<f64>::new().with_branching_factor(1);
        assert!(model.fit(&x, &()).is_err());
    }

    #[test]
    fn test_invalid_n_clusters() {
        let x = make_two_blobs();
        let model = Birch::<f64>::new().with_n_clusters(0);
        assert!(model.fit(&x, &()).is_err());
    }

    #[test]
    fn test_f32_support() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                0.0f32, 0.0, 0.1, 0.0, 0.0, 0.1, 10.0, 10.0, 10.1, 10.0, 10.0, 10.1,
            ],
        )
        .unwrap();

        let model = Birch::<f32>::new().with_threshold(0.5).with_n_clusters(2);
        let fitted = model.fit(&x, &()).unwrap();
        assert_eq!(fitted.labels().len(), 6);
    }

    #[test]
    fn test_labels_in_valid_range() {
        let x = make_three_blobs();
        let model = Birch::<f64>::new().with_threshold(0.5).with_n_clusters(3);
        let fitted = model.fit(&x, &()).unwrap();

        for &label in fitted.labels() {
            assert!(label < 3, "label {label} out of range [0, 3)");
        }
    }

    #[test]
    fn test_identical_points() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).unwrap();

        let model = Birch::<f64>::new().with_threshold(0.5).with_n_clusters(1);
        let fitted = model.fit(&x, &()).unwrap();

        for &label in fitted.labels() {
            assert_eq!(label, 0);
        }
    }

    #[test]
    fn test_default_constructor() {
        let model = Birch::<f64>::default();
        assert!(model.threshold > 0.0);
        assert_eq!(model.branching_factor, 50);
        assert!(model.n_clusters.is_none());
    }

    #[test]
    fn test_single_cluster() {
        let x = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.01, 0.0, 0.0, 0.01, 0.01, 0.01])
            .unwrap();

        let model = Birch::<f64>::new().with_threshold(1.0).with_n_clusters(1);
        let fitted = model.fit(&x, &()).unwrap();

        for &label in fitted.labels() {
            assert_eq!(label, 0);
        }
    }
}
