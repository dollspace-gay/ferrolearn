//! Agglomerative (bottom-up) hierarchical clustering.
//!
//! This module provides [`AgglomerativeClustering`], a hierarchical clustering
//! algorithm that builds a dendrogram by successively merging the two closest
//! clusters.  The merge criterion is determined by the [`Linkage`] strategy.
//!
//! # Algorithm
//!
//! This is a bit-exact translation of scikit-learn's *unstructured*
//! (`connectivity=None`) path, which delegates the dendrogram construction to
//! SciPy (`sklearn/cluster/_agglomerative.py:298-321, 532-592`):
//!
//! * **Ward / Complete / Average** use the **nearest-neighbour chain**
//!   (`nn-chain`) algorithm (`scipy.cluster.hierarchy.linkage`), producing a set
//!   of `n-1` merges. The merges are then **stably sorted by merge distance** and
//!   relabelled through a **union-find** so the resulting `children_` matches
//!   `scipy.cluster.hierarchy.linkage(X, method, 'euclidean')[:, :2]` bit-for-bit.
//! * **Single** uses **Prim's MST** (`mst_linkage_core`,
//!   `_hierarchical_fast.pyx`) sorted by weight, then a union-find relabel
//!   (`_single_linkage_label`) that — unlike the other linkages — does **not**
//!   reorder the merged pair.
//!
//! The full dendrogram has node IDs `0..n-1` for the leaves and `n+i` for the
//! cluster formed by the `i`-th sorted merge. `children_` therefore has shape
//! `(n_samples - 1, 2)` regardless of `n_clusters` (the FULL tree), exactly like
//! sklearn's fitted `children_` attribute.
//!
//! `labels_` are then produced by cutting the full tree with
//! [`_hc_cut`](https://github.com/scikit-learn/scikit-learn/blob/1.5.2/sklearn/cluster/_agglomerative.py#L731):
//! a negated-id max-heap pops the top `n_clusters` dendrogram nodes and numbers
//! each leaf by the heap position of its ancestor.
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
//! sklearn 1.5.2 oracle (R-CHAR-3). The `labels_` PARTITION (REQ-1), the full
//! `children_` dendrogram (REQ-6) and the absolute `_hc_cut` `labels_` numbering
//! (REQ-7) all SHIP bit-exact against scipy/sklearn 1.5.2 for the unstructured
//! (`connectivity=None`) case across all four linkages (the #938 structural
//! carve-out), via the nn-chain / MST dendrogram builder + `_hc_cut`. The
//! consumers `birch.rs` / `feature_agglomeration.rs` use `labels()` purely as a
//! partition and are unaffected by the (now sklearn-exact) renumbering.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`labels_` PARTITION, separable data) | SHIPPED | impl `fn agglomerate` (`fn full_dendrogram` → `fn hc_cut`) → `Fit::fit` builds `labels_`, mirroring sklearn `_fit` (`_agglomerative.py:1083-1105`). Now ABSOLUTE-value-matches the oracle for all 4 linkages (was up-to-permutation; see REQ-7). Consumers: `_RsAgglomerativeClustering` (`ferrolearn-python/src/extras.rs`), `birch.rs fn fit`, `feature_agglomeration.rs fn fit`. Guards: `green_two_blobs_partition_all_linkages`, `green_three_blobs_partition_all_linkages` in `tests/divergence_agglomerative.rs`. |
//! | REQ-2 (`n_clusters_` == requested) | SHIPPED | `Fit::fit` sets `n_clusters_: self.n_clusters`, mirroring `self.n_clusters_ = self.n_clusters` when `distance_threshold is None` (`_agglomerative.py:1095`). Guards above assert `n_clusters_` 2 and 3; `n_clusters()` accessor consumed by `birch.rs`/`feature_agglomeration.rs`. |
//! | REQ-3 (four linkage criteria) | SHIPPED | `enum Linkage` + `fn lance_williams` (Single=min, Complete=max, Average=size-weighted mean, Ward=Ward update on Euclidean distances) in the `fn nn_chain` builder, with `fn mst_single`+`fn single_linkage_relabel` for single, mirror `_TREE_BUILDERS` (`_agglomerative.py:720-725`). Now the `children_`/`labels_` BIT-EXACT-match the oracle for all four (REQ-6/REQ-7), not just the partition. Guards above + the parity tests in REQ-6/REQ-7. |
//! | REQ-4 (`n_clusters=2` ctor default + sklearn error ABI) | NOT-STARTED | open prereq blocker #963. sklearn `__init__` defaults `n_clusters=2` (`_agglomerative.py:951`); ferrolearn `fn new(n_clusters)` requires it. Validation errors are `FerroError::InvalidParameter`/`InsufficientSamples` (crate-wide port convention), not sklearn's `ValueError`/`InvalidParameterError`. |
//! | REQ-5 (`ensure_min_samples=2` validation) | NOT-STARTED | open prereq blocker #964. sklearn `fit` → `_validate_data(X, ensure_min_samples=2)` (`_agglomerative.py:989`) rejects `n_samples < 2`; ferrolearn `fn fit` accepts a single sample when `n_clusters <= 1` (`test_single_sample_single_cluster`). Coupled fix: `birch.rs fn fit` calls `AgglomerativeClustering::new(1)` on a 1-row matrix in the single-subcluster path, so this is a multi-file change, not minimal in `agglomerative.rs` alone. |
//! | REQ-6 (`children_` full-dendrogram format) | SHIPPED | impl `fn full_dendrogram in agglomerative.rs` (nn-chain `fn nn_chain` + stable distance sort + union-find `fn union_find_relabel` for ward/complete/average; Prim MST `fn mst_single` + `fn single_linkage_relabel` for single) produces `children_` of shape `(n_samples-1, 2)` with leaves `0..n-1` and internal-node IDs `n+i`, BIT-EXACT-equal to `scipy.cluster.hierarchy.linkage(X, method, 'euclidean')[:, :2]` for ward/complete/average (`_agglomerative.py:314`/`:586`); for `single`, sklearn uses `mst_linkage_core` + `_single_linkage_label` (`:567-584`, R-DEV-7) whose pair order differs from `scipy.linkage('single')`, so `children_` matches sklearn's OWN `AgglomerativeClustering.children_` bit-exact. Live-oracle tests in `tests/divergence_agglomerative_dendrogram.rs`: `children_exact_scipy_6pt_nn_chain_linkages`, `children_exact_scipy_10pt_nn_chain_linkages`, `children_exact_sklearn_single_6pt`, `children_exact_sklearn_single_10pt`, and the pinned `divergence_children_full_dendrogram_format`. Consumers: `Fit::fit` → `children_`, `RsAgglomerativeClustering`, `birch.rs`/`feature_agglomeration.rs`. |
//! | REQ-7 (`labels_` ABSOLUTE numbering via `_hc_cut`) | SHIPPED | impl `fn hc_cut in agglomerative.rs` (negated-id max-heap over the top-`n_clusters` dendrogram nodes + `fn hc_get_descendent`, mirroring `_hc_cut`, `_agglomerative.py:731-775`) builds `labels_` from the REQ-6 full `children_`, BIT-EXACT-equal to `sklearn.cluster.AgglomerativeClustering(n_clusters=k, linkage=…).fit(X).labels_`. Live-oracle tests: `labels_exact_sklearn_6pt_all_linkages`/`labels_exact_sklearn_10pt_all_linkages` (k∈{2,3}, all four linkages) + the pinned `divergence_labels_absolute_hc_cut_numbering`. Consumer: `Fit::fit` → `labels_` surfaced through `RsAgglomerativeClustering::labels_` + `birch.rs`/`feature_agglomeration.rs` (partition use). |
//! | REQ-8 (`metric` / `connectivity`) | NOT-STARTED | open prereq blocker #965. sklearn `metric` ∈ {euclidean,l1,l2,manhattan,cosine,precomputed} default `'euclidean'` with the ward-requires-euclidean rule (`_agglomerative.py:795-799`, `:1034-1038`) and `connectivity` for structured clustering (`:812-822`). ferrolearn `fn sq_euclidean`/`fn pairwise_sq_dists` are Euclidean-only, unstructured. |
//! | REQ-9 (`distance_threshold`/`compute_distances`/`distances_`) | SHIPPED | ctor `n_clusters: Option<usize>` (default `Some(2)`) + `distance_threshold: Option<F>` + `with_distance_threshold` (clears `n_clusters`, mirroring `n_clusters=None,distance_threshold=t`) + `with_compute_distances`; `Fit::fit` enforces the XOR `not ((n_clusters is None) ^ (distance_threshold is None))` (`_agglomerative.py:1022-1027`) as `FerroError::InvalidParameter`. `fn full_dendrogram`/`fn union_find_relabel`/`fn single_linkage_relabel` now also return the per-merge distances in `children_` row order; `fn agglomerate` surfaces them as `distances_` when `return_distance = distance_threshold.is_some() || compute_distances` (`:1074`, `:1087-1088`) and derives `n_clusters_ = count(distances_ >= t) + 1` (`:1090-1093`) then `labels_ = hc_cut(n_clusters_, …)` (`:1099`). `distances_` EXACT-equals `scipy.linkage(X, L, 'euclidean')[:, 2]` / sklearn `distances_` (~1e-12, all 4 linkages). Live-oracle tests `tests/divergence_agglomerative_threshold.rs`: `distances_exact_6pt_all_linkages`, `distances_exact_10pt_all_linkages`, `threshold_cut_exact_6pt_all_linkages`, `threshold_cut_exact_10pt_all_linkages`, `threshold_inclusive_boundary_6pt_ward`, `xor_*`. Consumers: `Fit::fit` → `distances_`/`n_clusters_`/`labels_`, `RsAgglomerativeClustering`, `birch.rs`/`feature_agglomeration.rs` (via `new()` default `Some(2)`). NOTE `compute_full_tree` is a no-op here: the unstructured path always builds the FULL dendrogram, so the `'auto'`/partial-tree PERF optimisation is unimplemented and not observable — every observable attribute matches the full-tree path. `memory` caching stays NOT-STARTED. |
//! | REQ-10 (`n_leaves_`/`n_connected_components_`) | SHIPPED | `FittedAgglomerativeClustering` now stores `pub n_leaves_: usize` / `pub n_connected_components_: usize` with accessors `n_leaves()`/`n_connected_components()`; `Fit::fit` sets `n_leaves_ = n_samples`, `n_connected_components_ = 1` for the unstructured (`connectivity=None`) path (`_agglomerative.py:1083-1085`: `ward_tree`/`linkage_tree` return `(children_, 1, n_samples, None[, distances])`). Live-oracle test `n_leaves_and_connected_components_unstructured` (`tests/divergence_agglomerative_threshold.rs`, both fixtures, all 4 linkages). Consumer: `Fit::fit` (production path) + the accessors. `memory` caching (`:1006`/`:1076`) stays NOT-STARTED (open blocker #967): it is an opt-in perf/persistence wrapper with no observable attribute divergence in the default `memory=None` path. |
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
/// # Cluster target: `n_clusters` XOR `distance_threshold`
///
/// Mirroring sklearn's `__init__` (`_agglomerative.py:949-960`), the number of
/// clusters is expressed as EITHER a fixed `n_clusters` OR a
/// `distance_threshold` (the linkage distance above which clusters are not
/// merged). Exactly one must be set — sklearn enforces this as the XOR
/// `not ((n_clusters is None) ^ (distance_threshold is None))`
/// (`_agglomerative.py:1022-1027`). Rust has no `None`-vs-`int` overload, so we
/// store `n_clusters: Option<usize>` (default `Some(2)`, mirroring sklearn's
/// `n_clusters=2` default) and `distance_threshold: Option<F>` (default
/// `None`). [`with_distance_threshold`](Self::with_distance_threshold) sets the
/// threshold and clears `n_clusters` to mirror the sklearn idiom
/// `AgglomerativeClustering(n_clusters=None, distance_threshold=t)`.
///
/// # Type Parameters
///
/// - `F`: floating-point scalar type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct AgglomerativeClustering<F> {
    /// Target number of clusters. `None` when [`distance_threshold`] drives the
    /// cut instead (mirrors sklearn `n_clusters=None`).
    ///
    /// [`distance_threshold`]: Self::distance_threshold
    pub n_clusters: Option<usize>,
    /// Linkage strategy for computing inter-cluster distances.
    pub linkage: Linkage,
    /// The linkage distance threshold at or above which clusters will not be
    /// merged. `None` when [`n_clusters`] drives the cut (mirrors sklearn
    /// `distance_threshold=None`).
    ///
    /// [`n_clusters`]: Self::n_clusters
    pub distance_threshold: Option<F>,
    /// Whether to compute and store the per-merge `distances_` even when
    /// `distance_threshold` is not set (mirrors sklearn `compute_distances`,
    /// default `false`). `distances_` is always computed when
    /// `distance_threshold` is set (`return_distance`, `_agglomerative.py:1074`).
    pub compute_distances: bool,
    /// Phantom to retain the float type parameter.
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float> AgglomerativeClustering<F> {
    /// Create a new `AgglomerativeClustering` with the given number of clusters.
    ///
    /// Uses default `linkage = Ward`, `distance_threshold = None`,
    /// `compute_distances = false`.
    #[must_use]
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters: Some(n_clusters),
            linkage: Linkage::Ward,
            distance_threshold: None,
            compute_distances: false,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the linkage criterion.
    #[must_use]
    pub fn with_linkage(mut self, linkage: Linkage) -> Self {
        self.linkage = linkage;
        self
    }

    /// Drive the cut by a linkage-distance threshold instead of a fixed cluster
    /// count, mirroring sklearn
    /// `AgglomerativeClustering(n_clusters=None, distance_threshold=t)`.
    ///
    /// Sets `distance_threshold = Some(threshold)` and clears `n_clusters` to
    /// `None` (the XOR contract, `_agglomerative.py:1022-1027`). After fitting,
    /// `n_clusters_ = count(distances_ >= threshold) + 1`
    /// (`_agglomerative.py:1090-1093`). Setting the threshold also forces
    /// `distances_` to be computed (`return_distance`, `:1074`).
    #[must_use]
    pub fn with_distance_threshold(mut self, threshold: F) -> Self {
        self.distance_threshold = Some(threshold);
        self.n_clusters = None;
        self
    }

    /// Compute and store the per-merge `distances_` array even when no
    /// `distance_threshold` is set, mirroring sklearn `compute_distances=True`
    /// (`_agglomerative.py:959`, `:1074`, `:1087-1088`).
    #[must_use]
    pub fn with_compute_distances(mut self, compute_distances: bool) -> Self {
        self.compute_distances = compute_distances;
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
    /// The full dendrogram, shape `(n_samples - 1, 2)`.
    ///
    /// Each element `(a, b)` records that nodes `a` and `b` were merged to form
    /// node `n_samples + i` (where `i` is the row index). Values `< n_samples`
    /// are leaves (original samples); values `>= n_samples` are internal nodes.
    /// This matches scikit-learn's `children_` attribute and
    /// `scipy.cluster.hierarchy.linkage(X, method, 'euclidean')[:, :2]`.
    pub children_: Vec<(usize, usize)>,
    /// The per-merge linkage distances, in `children_` row order, length
    /// `n_samples - 1`. `Some` when `distance_threshold` was set OR
    /// `compute_distances` was `true` (mirroring sklearn's `return_distance`
    /// path, `_agglomerative.py:1074`, `:1087-1088`), `None` otherwise.
    ///
    /// Equal to `scipy.cluster.hierarchy.linkage(X, method, 'euclidean')[:, 2]`
    /// for ward/complete/average and to sklearn's own `distances_` for single.
    pub distances_: Option<Array1<F>>,
    /// The number of leaves in the hierarchical tree, equal to `n_samples` for
    /// the unstructured (`connectivity=None`) path
    /// (`_agglomerative.py:1083-1085`).
    pub n_leaves_: usize,
    /// The estimated number of connected components in the graph. Always `1` for
    /// the unstructured (`connectivity=None`) path
    /// (`_agglomerative.py:1083-1085`).
    pub n_connected_components_: usize,
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

    /// Return the full dendrogram, shape `(n_samples - 1, 2)`.
    ///
    /// Row `i` is the pair of node IDs merged to form node `n_samples + i`
    /// (leaves are `0..n_samples`, internal nodes are `>= n_samples`). This
    /// matches scikit-learn's `children_` /
    /// `scipy.cluster.hierarchy.linkage(...)[:, :2]`.
    #[must_use]
    pub fn children(&self) -> &[(usize, usize)] {
        &self.children_
    }

    /// Return the per-merge linkage distances (in `children_` row order), or
    /// `None` if they were not computed (neither `distance_threshold` nor
    /// `compute_distances` was set). Mirrors sklearn's `distances_` attribute
    /// (`_agglomerative.py:1087-1088`).
    #[must_use]
    pub fn distances(&self) -> Option<&Array1<F>> {
        self.distances_.as_ref()
    }

    /// Return the number of leaves in the hierarchical tree (`= n_samples` for
    /// the unstructured path). Mirrors sklearn's `n_leaves_`
    /// (`_agglomerative.py:1083-1085`).
    #[must_use]
    pub fn n_leaves(&self) -> usize {
        self.n_leaves_
    }

    /// Return the number of connected components (`= 1` for the unstructured
    /// path). Mirrors sklearn's `n_connected_components_`
    /// (`_agglomerative.py:1083-1085`).
    #[must_use]
    pub fn n_connected_components(&self) -> usize {
        self.n_connected_components_
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the *Euclidean* distance between two row slices.
///
/// scipy's `linkage`/`ward` operate on actual (non-squared) Euclidean distances;
/// the Ward Lance–Williams update below works on these directly.
#[inline]
fn euclidean<F: Float>(a: &[F], b: &[F]) -> f64 {
    a.iter()
        .zip(b.iter())
        .fold(0.0_f64, |acc, (&ai, &bi)| {
            let d = ai.to_f64().unwrap_or(0.0) - bi.to_f64().unwrap_or(0.0);
            acc + d * d
        })
        .sqrt()
}

/// Index into the condensed (upper-triangular) distance vector for the pair
/// `(i, j)` over `n` observations, mirroring scipy's `condensed_index`.
#[inline]
fn condensed_index(n: usize, i: usize, j: usize) -> usize {
    if i < j {
        n * i - (i * (i + 1)) / 2 + (j - i - 1)
    } else {
        n * j - (j * (j + 1)) / 2 + (i - j - 1)
    }
}

/// Lance–Williams distance update for the nn-chain linkages, matching scipy's
/// `linkage_distance_update` family (operating on Euclidean distances).
#[inline]
fn lance_williams(
    linkage: Linkage,
    d_xi: f64,
    d_yi: f64,
    size_x: f64,
    size_y: f64,
    size_i: f64,
    d_xy: f64,
) -> f64 {
    match linkage {
        Linkage::Single => d_xi.min(d_yi),
        Linkage::Complete => d_xi.max(d_yi),
        Linkage::Average => (size_x * d_xi + size_y * d_yi) / (size_x + size_y),
        Linkage::Ward => {
            let t = 1.0 / (size_x + size_y + size_i);
            ((size_i + size_x) * t * d_xi * d_xi + (size_i + size_y) * t * d_yi * d_yi
                - size_i * t * d_xy * d_xy)
                .sqrt()
        }
    }
}

/// A single merge produced by the chain/MST builders, before relabelling.
/// `(node_a, node_b, distance)` with the original observation indices.
type RawMerge = (usize, usize, f64);

/// Nearest-neighbour-chain dendrogram builder for ward / complete / average,
/// mirroring scipy's `nn_chain` (`scipy/cluster/_hierarchy.pyx`). Returns the
/// `n-1` merges (unordered) as `(x, y, dist)` with `x < y` original-index roots
/// at merge time — exactly scipy's `Z[:, :3]` before the final sort/relabel.
fn nn_chain(condensed: &[f64], n: usize, linkage: Linkage) -> Vec<RawMerge> {
    let mut d = condensed.to_vec();
    let mut size = vec![1.0_f64; n];
    let mut chain: Vec<usize> = vec![0; n];
    let mut chain_len = 0usize;
    let mut merges: Vec<RawMerge> = Vec::with_capacity(n.saturating_sub(1));

    for _ in 0..n.saturating_sub(1) {
        if chain_len == 0 {
            chain_len = 1;
            // First active cluster (size > 0).
            for (i, &s) in size.iter().enumerate() {
                if s > 0.0 {
                    chain[0] = i;
                    break;
                }
            }
        }

        let mut x;
        let mut y;
        let mut current_min;
        loop {
            x = chain[chain_len - 1];
            if chain_len > 1 {
                y = chain[chain_len - 2];
                current_min = d[condensed_index(n, x, y)];
            } else {
                y = usize::MAX;
                current_min = f64::INFINITY;
            }
            for (i, &si) in size.iter().enumerate() {
                if si == 0.0 || x == i {
                    continue;
                }
                let dist = d[condensed_index(n, x, i)];
                if dist < current_min {
                    current_min = dist;
                    y = i;
                }
            }
            if chain_len > 1 && y == chain[chain_len - 2] {
                break;
            }
            chain[chain_len] = y;
            chain_len += 1;
        }

        chain_len -= 2;

        // Merge x and y; ensure x < y (scipy's convention).
        if x > y {
            std::mem::swap(&mut x, &mut y);
        }
        let nx = size[x];
        let ny = size[y];
        merges.push((x, y, current_min));

        size[x] = 0.0;
        size[y] = nx + ny;

        // Lance–Williams update of distances to the merged cluster (stored in y).
        for i in 0..n {
            let ni = size[i];
            if ni == 0.0 || i == y {
                continue;
            }
            let d_ix = d[condensed_index(n, i, x)];
            let d_iy = d[condensed_index(n, i, y)];
            d[condensed_index(n, i, y)] =
                lance_williams(linkage, d_ix, d_iy, nx, ny, ni, current_min);
        }
    }

    merges
}

/// Prim's MST builder for single linkage, mirroring sklearn's
/// `mst_linkage_core` (`_hierarchical_fast.pyx`). Distances are generated on the
/// fly (Euclidean). Returns `n-1` merges as `(current_node, new_node, dist)` in
/// MST-construction order (NOT sorted).
fn mst_single<F: Float>(x: &Array2<F>, n: usize) -> Vec<RawMerge> {
    let rows: Vec<&[F]> = (0..n).map(|i| row_slice(x, i)).collect();
    let mut in_tree = vec![false; n];
    let mut current_distances = vec![f64::INFINITY; n];
    let mut current_node = 0usize;
    let mut merges: Vec<RawMerge> = Vec::with_capacity(n.saturating_sub(1));

    for _ in 0..n.saturating_sub(1) {
        in_tree[current_node] = true;
        let mut new_distance = f64::INFINITY;
        let mut new_node = 0usize;

        for j in 0..n {
            if in_tree[j] {
                continue;
            }
            let left_value = euclidean(rows[current_node], rows[j]);
            if left_value < current_distances[j] {
                current_distances[j] = left_value;
            }
            if current_distances[j] < new_distance {
                new_distance = current_distances[j];
                new_node = j;
            }
        }

        merges.push((current_node, new_node, new_distance));
        current_node = new_node;
    }

    merges
}

/// The full dendrogram plus the per-merge distances in `children_` row order.
/// `distances[i]` is the linkage distance at which `children[i]` merged — the
/// 3rd column of the scipy linkage matrix / sklearn's `distances_`
/// (`_agglomerative.py:1087-1088`).
type Dendrogram = (Vec<(usize, usize)>, Vec<f64>);

/// Union-find relabel for the nn-chain linkages, mirroring scipy's `label`
/// (`scipy/cluster/_hierarchy.pyx`). Merges must already be **stably sorted by
/// distance**. Emits the full dendrogram with the smaller root first, plus the
/// per-merge distances in that same (sorted) row order — `scipy.linkage(...)[:, 2]`.
fn union_find_relabel(mut merges: Vec<RawMerge>, n: usize) -> Dendrogram {
    // Stable sort by merge distance (scipy uses `argsort(kind='stable')`).
    merges.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

    let mut parent: Vec<usize> = (0..(2 * n).saturating_sub(1)).collect();
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(n.saturating_sub(1));
    let mut dists: Vec<f64> = Vec::with_capacity(n.saturating_sub(1));

    for (i, (a, b, dist)) in merges.into_iter().enumerate() {
        let next_label = n + i;
        let ra = uf_find(&mut parent, a);
        let rb = uf_find(&mut parent, b);
        let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
        out.push((lo, hi));
        dists.push(dist);
        parent[ra] = next_label;
        parent[rb] = next_label;
    }

    (out, dists)
}

/// Union-find relabel for single linkage, mirroring sklearn's
/// `_single_linkage_label` + `UnionFind` (`_hierarchical_fast.pyx`). The MST must
/// already be **stably sorted by weight**. Unlike [`union_find_relabel`], the
/// pair is emitted in `(left_root, right_root)` order WITHOUT reordering. The
/// per-merge distances are returned in that same (sorted) row order, equal to
/// sklearn's `distances_`.
fn single_linkage_relabel(mut merges: Vec<RawMerge>, n: usize) -> Dendrogram {
    merges.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

    let mut parent: Vec<usize> = (0..(2 * n).saturating_sub(1)).collect();
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(n.saturating_sub(1));
    let mut dists: Vec<f64> = Vec::with_capacity(n.saturating_sub(1));

    for (i, (left, right, dist)) in merges.into_iter().enumerate() {
        let next_label = n + i;
        let lc = uf_find(&mut parent, left);
        let rc = uf_find(&mut parent, right);
        out.push((lc, rc));
        dists.push(dist);
        parent[lc] = next_label;
        parent[rc] = next_label;
    }

    (out, dists)
}

/// Find with path compression over a parent array where a self-parent denotes a
/// root. (Matches the union-find semantics of scipy `label` / sklearn
/// `UnionFind.fast_find`.)
fn uf_find(parent: &mut [usize], mut node: usize) -> usize {
    let mut root = node;
    while parent[root] != root {
        root = parent[root];
    }
    // Path compression.
    while parent[node] != root {
        let next = parent[node];
        parent[node] = root;
        node = next;
    }
    root
}

/// Collect a contiguous row slice for sample `i`, falling back to a copy when
/// the row is not contiguous.
fn row_slice<F: Float>(x: &Array2<F>, i: usize) -> &[F] {
    x.row(i).to_slice().unwrap_or(&[])
}

/// Build the FULL dendrogram (shape `(n_samples - 1, 2)`) for the given linkage,
/// bit-exact with `scipy.cluster.hierarchy.linkage(X, method, 'euclidean')[:, :2]`
/// (ward == `method='ward'`). Leaves are `0..n`, the `i`-th sorted merge forms
/// node `n + i`. Also returns the per-merge distances in `children_` row order
/// (the 3rd column of the scipy linkage matrix / sklearn's `distances_`).
fn full_dendrogram<F: Float>(x: &Array2<F>, linkage: Linkage) -> Dendrogram {
    let n = x.nrows();
    if n < 2 {
        return (Vec::new(), Vec::new());
    }

    match linkage {
        Linkage::Single => {
            let merges = mst_single(x, n);
            single_linkage_relabel(merges, n)
        }
        _ => {
            // Build the condensed Euclidean distance vector.
            let rows: Vec<&[F]> = (0..n).map(|i| row_slice(x, i)).collect();
            let mut condensed = vec![0.0_f64; n * (n - 1) / 2];
            let mut k = 0;
            for i in 0..n {
                for j in (i + 1)..n {
                    condensed[k] = euclidean(rows[i], rows[j]);
                    k += 1;
                }
            }
            let merges = nn_chain(&condensed, n, linkage);
            union_find_relabel(merges, n)
        }
    }
}

/// Collect all descendant leaves of `node` in the dendrogram, mirroring
/// sklearn's `_hc_get_descendent` (`_hierarchical_fast.pyx`). Leaves are
/// `< n_leaves`; an internal node `i` has children `children[i - n_leaves]`.
fn hc_get_descendent(node: usize, children: &[(usize, usize)], n_leaves: usize) -> Vec<usize> {
    if node < n_leaves {
        return vec![node];
    }
    let mut stack = vec![node];
    let mut descendents = Vec::new();
    while let Some(i) = stack.pop() {
        if i < n_leaves {
            descendents.push(i);
        } else {
            let (a, b) = children[i - n_leaves];
            stack.push(a);
            stack.push(b);
        }
    }
    descendents
}

/// Cut the full dendrogram into `n_clusters` clusters, numbering each leaf by
/// the heap position of its top-level ancestor — a faithful, operation-for-
/// operation translation of sklearn's `_hc_cut` (`_agglomerative.py:731-775`).
///
/// `nodes` stores NEGATED node ids in a Python-`heapq` MIN-heap (so the smallest
/// element is the largest node id). We replicate `heappush` and `heappushpop`
/// exactly, then `enumerate` the heap array in its final layout order — both the
/// heap-array layout and the enumeration order are reproduced verbatim, so the
/// absolute `labels_` match sklearn bit-for-bit.
fn hc_cut(n_clusters: usize, children: &[(usize, usize)], n_leaves: usize) -> Array1<usize> {
    let mut labels = Array1::<usize>::zeros(n_leaves);
    if n_leaves == 0 || children.is_empty() {
        // Single leaf (or no merges) → one cluster (label 0 everywhere).
        return labels;
    }

    // nodes = [-(max(children[-1]) + 1)]  (negated root id).
    let last = children[children.len() - 1];
    let root = (last.0.max(last.1) + 1) as i64;
    let mut nodes: Vec<i64> = vec![-root];

    for _ in 0..n_clusters.saturating_sub(1) {
        // these_children = children[-nodes[0] - n_leaves]; nodes[0] is the
        // min (= most-negative) element of the Python heap.
        let smallest = nodes[0]; // negated => smallest negated == largest id
        let node_id = (-smallest) as usize;
        let these = children[node_id - n_leaves];
        // heappush(nodes, -these[0]); heappushpop(nodes, -these[1]);
        heappush_min(&mut nodes, -(these.0 as i64));
        heappushpop_min(&mut nodes, -(these.1 as i64));
    }

    // for i, node in enumerate(nodes): label[descendents(-node)] = i
    for (i, &neg_node) in nodes.iter().enumerate() {
        let node = (-neg_node) as usize;
        for leaf in hc_get_descendent(node, children, n_leaves) {
            labels[leaf] = i;
        }
    }
    labels
}

/// CPython `heapq.heappush` (sift-down of the new last element toward the root),
/// reproducing the exact array layout so `_hc_cut`'s `enumerate(nodes)` order
/// matches.
fn heappush_min(heap: &mut Vec<i64>, item: i64) {
    heap.push(item);
    let last = heap.len() - 1;
    sift_down(heap, 0, last);
}

/// CPython `heapq.heappushpop`: push `item`, then pop-and-return the smallest.
/// If `item` is no larger than the current min, it is returned unchanged
/// (matching CPython's fast path) and the heap is untouched.
fn heappushpop_min(heap: &mut [i64], item: i64) -> i64 {
    if !heap.is_empty() && heap[0] < item {
        let returned = heap[0];
        heap[0] = item;
        sift_up(heap, 0);
        return returned;
    }
    item
}

/// CPython `heapq._siftdown(heap, startpos, pos)`: bubble `heap[pos]` up toward
/// `startpos` while it is smaller than its parent.
fn sift_down(heap: &mut [i64], startpos: usize, pos: usize) {
    let mut pos = pos;
    let new_item = heap[pos];
    while pos > startpos {
        let parentpos = (pos - 1) >> 1;
        let parent = heap[parentpos];
        if new_item < parent {
            heap[pos] = parent;
            pos = parentpos;
        } else {
            break;
        }
    }
    heap[pos] = new_item;
}

/// CPython `heapq._siftup(heap, pos)`: move `heap[pos]` down to a leaf along the
/// path of smaller children, then sift it back up to its correct spot. This is
/// CPython's exact (non-obvious) layout-preserving variant.
fn sift_up(heap: &mut [i64], pos: usize) {
    let endpos = heap.len();
    let startpos = pos;
    let mut pos = pos;
    let new_item = heap[pos];
    let mut childpos = 2 * pos + 1;
    while childpos < endpos {
        let rightpos = childpos + 1;
        if rightpos < endpos && heap[childpos] >= heap[rightpos] {
            childpos = rightpos;
        }
        heap[pos] = heap[childpos];
        pos = childpos;
        childpos = 2 * pos + 1;
    }
    heap[pos] = new_item;
    sift_down(heap, startpos, pos);
}

/// The realised cut: `(labels_, children_, n_clusters_, distances_)`.
/// `distances_` is `Some` iff the distances were requested (`return_distance`).
type AgglomerateResult<F> =
    Result<(Array1<usize>, Vec<(usize, usize)>, usize, Option<Array1<F>>), FerroError>;

/// Run the full unstructured agglomerative pipeline for `x`, mirroring sklearn's
/// `_fit` (`_agglomerative.py:1066-1106`) for the `connectivity=None` case:
///
/// 1. build the full dendrogram (`children_` + per-merge `distances_`);
/// 2. `return_distance = (distance_threshold is not None) or compute_distances`
///    (`:1074`) decides whether `distances_` is surfaced;
/// 3. `n_clusters_` is either `count(distances_ >= threshold) + 1`
///    (`:1090-1093`) or the requested `n_clusters` (`:1095`);
/// 4. `labels_ = _hc_cut(n_clusters_, children_, n_leaves)` (`:1099`).
fn agglomerate<F: Float>(
    x: &Array2<F>,
    n_clusters: Option<usize>,
    distance_threshold: Option<F>,
    compute_distances: bool,
    linkage: Linkage,
) -> AgglomerateResult<F> {
    let n_samples = x.nrows();
    let (children, raw_dists) = full_dendrogram(x, linkage);

    // return_distance (`_agglomerative.py:1074`).
    let return_distance = distance_threshold.is_some() || compute_distances;

    // Convert the per-merge distances to F (computed in f64 internally).
    let distances_f: Vec<F> = raw_dists
        .iter()
        .map(|&d| F::from(d).unwrap_or_else(F::zero))
        .collect();

    // n_clusters_ (`_agglomerative.py:1090-1095`).
    let n_clusters_ = match distance_threshold {
        Some(t) => {
            // np.count_nonzero(distances_ >= distance_threshold) + 1.
            // sklearn's `distances_` and the threshold are float64 (scipy's
            // `linkage`/`ward_tree` return float64 heights regardless of
            // `X.dtype`), so count against the RAW f64 heights and an
            // f64-promoted threshold — NOT the F-downcast copy, whose f32
            // rounding of a merge height would flip the count near the
            // threshold (#2185).
            let t_f64 = t.to_f64().unwrap_or(f64::NAN);
            let count = raw_dists.iter().filter(|&&d| d >= t_f64).count();
            count + 1
        }
        None => n_clusters.unwrap_or(0),
    };

    let labels = hc_cut(n_clusters_, &children, n_samples);

    let distances_out = if return_distance {
        Some(Array1::from(distances_f))
    } else {
        None
    };

    Ok((labels, children, n_clusters_, distances_out))
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
    /// - [`FerroError::InvalidParameter`] (`"n_clusters"` / `"distance_threshold"`)
    ///   when neither or both of `n_clusters`/`distance_threshold` is set
    ///   (sklearn's XOR, `_agglomerative.py:1022-1027`), or when `n_clusters == 0`.
    /// - [`FerroError::InsufficientSamples`] if `n_samples == 0`, or (in the
    ///   fixed-`n_clusters` path) `n_samples < n_clusters`.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedAgglomerativeClustering<F>, FerroError> {
        // XOR: exactly one of n_clusters / distance_threshold must be set.
        // Mirrors `not ((n_clusters is None) ^ (distance_threshold is None))`
        // (`_agglomerative.py:1022-1027`).
        match (self.n_clusters, self.distance_threshold) {
            (Some(_), Some(_)) | (None, None) => {
                return Err(FerroError::InvalidParameter {
                    name: "distance_threshold".into(),
                    reason: "Exactly one of n_clusters and distance_threshold has to be set, \
                             and the other needs to be None"
                        .into(),
                });
            }
            _ => {}
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

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: self.n_clusters.unwrap_or(1),
                actual: 0,
                context: "AgglomerativeClustering requires at least n_clusters samples".into(),
            });
        }

        if let Some(k) = self.n_clusters
            && n_samples < k
        {
            return Err(FerroError::InsufficientSamples {
                required: k,
                actual: n_samples,
                context: "AgglomerativeClustering requires at least n_clusters samples".into(),
            });
        }

        let (labels, children, n_clusters_, distances) = agglomerate(
            x,
            self.n_clusters,
            self.distance_threshold,
            self.compute_distances,
            self.linkage,
        )?;

        Ok(FittedAgglomerativeClustering {
            labels_: labels,
            n_clusters_,
            children_: children,
            distances_: distances,
            // Unstructured (connectivity=None): n_leaves_ == n_samples,
            // n_connected_components_ == 1 (`_agglomerative.py:1083-1085`).
            n_leaves_: n_samples,
            n_connected_components_: 1,
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

    /// Three collinear points, well separated (infallible `arr2` fixture).
    fn three_points() -> Array2<f64> {
        ndarray::arr2(&[[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]])
    }

    // ── Construction ────────────────────────────────────────────────────────

    #[test]
    fn test_new_defaults() {
        let model = AgglomerativeClustering::<f64>::new(3);
        assert_eq!(model.n_clusters, Some(3));
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
        // Full dendrogram: children_ has n_samples - 1 rows regardless of
        // n_clusters (matches sklearn / scipy linkage output).
        let x = make_two_blobs(); // 8 samples → 7 dendrogram rows.
        let fitted = AgglomerativeClustering::<f64>::new(2).fit(&x, &());
        let fitted = fitted.as_ref().ok();
        assert!(fitted.is_some());
        if let Some(f) = fitted {
            assert_eq!(f.children().len(), x.nrows() - 1);
        }
    }

    #[test]
    fn test_children_full_tree_when_n_clusters_equals_n_samples() {
        // children_ is the FULL dendrogram (n_samples - 1 rows) even when
        // n_clusters == n_samples — sklearn always builds the full tree.
        let x = three_points();
        let fitted = AgglomerativeClustering::<f64>::new(3).fit(&x, &());
        let fitted = fitted.as_ref().ok();
        assert!(fitted.is_some());
        if let Some(f) = fitted {
            assert_eq!(f.children().len(), 2);
        }
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
