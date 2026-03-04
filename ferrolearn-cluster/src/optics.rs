//! OPTICS — Ordering Points To Identify the Clustering Structure.
//!
//! This module provides [`OPTICS`], a density-based algorithm that computes a
//! **reachability ordering** of the data.  Unlike DBSCAN, OPTICS does not
//! require a global density threshold; instead it produces a reachability plot
//! from which clusters at various density levels can be extracted.
//!
//! # Algorithm
//!
//! 1. For each unprocessed point `p`:
//!    - Compute its **core distance** — the distance to the `min_samples`-th
//!      nearest neighbour (within `max_eps`), or `∞` if there are fewer than
//!      `min_samples` neighbours.
//!    - If `p` is a core point, update the reachability distances of all
//!      unprocessed neighbours and add them to an ordered seed list.
//!    - Append `p` to the ordering with its final reachability distance.
//!
//! 2. **Cluster extraction** via the Xi method (see [`FittedOPTICS::extract_clusters`]):
//!    steep descents in the reachability plot define cluster boundaries.
//!
//! OPTICS does **not** implement [`Predict`](ferrolearn_core::Predict) — it
//! produces a reachability ordering and reachability distances from which
//! cluster memberships can be derived post-hoc.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_cluster::OPTICS;
//! use ferrolearn_core::Fit;
//! use ndarray::Array2;
//!
//! let x = Array2::from_shape_vec((9, 2), vec![
//!     0.0, 0.0,  0.1, 0.1,  0.0, 0.1,
//!     5.0, 5.0,  5.1, 5.0,  5.0, 5.1,
//!     10.0, 0.0, 10.1, 0.0, 10.0, 0.1,
//! ]).unwrap();
//!
//! let fitted = OPTICS::<f64>::new(2).fit(&x, &()).unwrap();
//! assert_eq!(fitted.ordering().len(), 9);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::Fit;
use ndarray::{Array1, Array2};
use num_traits::Float;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

// ─────────────────────────────────────────────────────────────────────────────
// Helper: ordered float wrapper for the priority queue
// ─────────────────────────────────────────────────────────────────────────────

/// A `(reachability_distance, point_index)` pair that forms a min-heap entry.
#[derive(Clone, Copy, PartialEq)]
struct SeedEntry<F: Float> {
    reach_dist: F,
    idx: usize,
}

impl<F: Float> Eq for SeedEntry<F> {}

/// We want a **min**-heap, so we reverse the comparison.
impl<F: Float> Ord for SeedEntry<F> {
    fn cmp(&self, other: &Self) -> Ordering {
        // NaN is treated as greater than anything (put last).
        other
            .reach_dist
            .partial_cmp(&self.reach_dist)
            .unwrap_or(Ordering::Less)
    }
}

impl<F: Float> PartialOrd for SeedEntry<F> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration struct
// ─────────────────────────────────────────────────────────────────────────────

/// OPTICS clustering configuration (unfitted).
///
/// Holds hyperparameters.  Call [`Fit::fit`] to run the algorithm and produce
/// a [`FittedOPTICS`].
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct OPTICS<F> {
    /// Minimum number of points required to form a core point (including
    /// the point itself).
    pub min_samples: usize,
    /// Maximum radius considered for neighbourhood queries.  Points beyond
    /// this distance are not considered neighbours.  Defaults to `F::infinity()`.
    pub max_eps: F,
    /// Xi steep-point threshold used by [`FittedOPTICS::extract_clusters`].
    /// A value in `(0, 1)`.  Defaults to `0.05`.
    pub xi: F,
}

impl<F: Float> OPTICS<F> {
    /// Create a new `OPTICS` with the given `min_samples`.
    ///
    /// Defaults: `max_eps = F::infinity()`, `xi = 0.05`.
    #[must_use]
    pub fn new(min_samples: usize) -> Self {
        Self {
            min_samples,
            max_eps: F::infinity(),
            xi: F::from(0.05).unwrap_or_else(|| F::from(5e-2).unwrap()),
        }
    }

    /// Set the maximum neighbourhood radius.
    #[must_use]
    pub fn with_max_eps(mut self, max_eps: F) -> Self {
        self.max_eps = max_eps;
        self
    }

    /// Set the Xi steep-point threshold.
    ///
    /// Must be in `(0, 1)`.
    #[must_use]
    pub fn with_xi(mut self, xi: F) -> Self {
        self.xi = xi;
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fitted struct
// ─────────────────────────────────────────────────────────────────────────────

/// Fitted OPTICS model.
///
/// Stores the reachability ordering, reachability distances, core distances,
/// and cluster labels (extracted via the Xi method).
///
/// OPTICS does **not** implement [`Predict`](ferrolearn_core::Predict).
#[derive(Debug, Clone)]
pub struct FittedOPTICS<F> {
    /// Indices of data points in the reachability order.
    ordering_: Vec<usize>,
    /// Reachability distance for each data point (indexed by original point
    /// index, not by ordering position).  The first point processed always
    /// has reachability distance `∞`.
    reachability_: Array1<F>,
    /// Core distance for each point (indexed by original point index).
    /// Equals `∞` for non-core points.
    core_distances_: Array1<F>,
    /// Cluster label for each training sample (0-indexed for clusters; `-1`
    /// for noise).  Extracted using the Xi method.
    labels_: Array1<isize>,
}

impl<F: Float> FittedOPTICS<F> {
    /// Return the reachability ordering (indices into the training data).
    #[must_use]
    pub fn ordering(&self) -> &[usize] {
        &self.ordering_
    }

    /// Return the reachability distances, indexed by original point index.
    #[must_use]
    pub fn reachability(&self) -> &Array1<F> {
        &self.reachability_
    }

    /// Return the core distances, indexed by original point index.
    #[must_use]
    pub fn core_distances(&self) -> &Array1<F> {
        &self.core_distances_
    }

    /// Return the cluster labels (Xi-method extraction).
    ///
    /// Noise points have label `-1`.
    #[must_use]
    pub fn labels(&self) -> &Array1<isize> {
        &self.labels_
    }

    /// Return the number of clusters found (excluding noise).
    #[must_use]
    pub fn n_clusters(&self) -> usize {
        let max_label = self.labels_.iter().max().copied().unwrap_or(-1);
        if max_label < 0 {
            0
        } else {
            (max_label + 1) as usize
        }
    }

    /// Extract flat clusters from the reachability plot using the Xi method.
    ///
    /// The Xi method identifies *steep up* and *steep down* areas in the
    /// reachability plot.  Clusters are formed between matching steep-down /
    /// steep-up pairs.
    ///
    /// `xi` must be in `(0, 1)`.  Returns a vector of cluster labels
    /// (length == `n_samples`); noise has label `-1`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if `xi` is outside `(0, 1)`.
    pub fn extract_clusters(&self, xi: F) -> Result<Array1<isize>, FerroError> {
        if xi <= F::zero() || xi >= F::one() {
            return Err(FerroError::InvalidParameter {
                name: "xi".into(),
                reason: "must be in (0, 1)".into(),
            });
        }
        Ok(xi_cluster_extraction(
            &self.ordering_,
            &self.reachability_,
            xi,
        ))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Euclidean distance between two slices.
#[inline]
fn euclidean<F: Float>(a: &[F], b: &[F]) -> F {
    a.iter()
        .zip(b)
        .fold(F::zero(), |acc, (&ai, &bi)| acc + (ai - bi) * (ai - bi))
        .sqrt()
}

/// Return all neighbours of `idx` within distance `max_eps` (sorted by distance).
///
/// Returns `(neighbor_indices, distances)` in ascending distance order.
fn get_neighbors<F: Float>(x: &Array2<F>, idx: usize, max_eps: F) -> (Vec<usize>, Vec<F>) {
    let row = x.row(idx);
    let rs = row.as_slice().unwrap_or(&[]);
    let mut pairs: Vec<(F, usize)> = (0..x.nrows())
        .filter_map(|j| {
            let other = x.row(j);
            let os = other.as_slice().unwrap_or(&[]);
            let d = euclidean(rs, os);
            if d <= max_eps && j != idx {
                Some((d, j))
            } else {
                None
            }
        })
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    let indices = pairs.iter().map(|p| p.1).collect();
    let dists = pairs.iter().map(|p| p.0).collect();
    (indices, dists)
}

/// Compute core distance of `idx`: distance to the `min_samples`-th nearest
/// neighbour within `max_eps`.  Returns `F::infinity()` if fewer than
/// `min_samples` neighbours exist.
fn core_distance<F: Float>(x: &Array2<F>, idx: usize, max_eps: F, min_samples: usize) -> F {
    let row = x.row(idx);
    let rs = row.as_slice().unwrap_or(&[]);

    let mut dists: Vec<F> = (0..x.nrows())
        .filter_map(|j| {
            if j == idx {
                return None;
            }
            let other = x.row(j);
            let os = other.as_slice().unwrap_or(&[]);
            let d = euclidean(rs, os);
            if d <= max_eps { Some(d) } else { None }
        })
        .collect();

    if dists.len() < min_samples.saturating_sub(1) {
        // Not enough neighbours (need min_samples - 1 others).
        return F::infinity();
    }

    dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    // The core distance is the distance to the (min_samples-1)-th other point
    // (0-indexed), i.e., the min_samples-th point overall including self.
    let k = min_samples.saturating_sub(1);
    if k == 0 {
        F::zero()
    } else if k <= dists.len() {
        dists[k - 1]
    } else {
        F::infinity()
    }
}

/// Update the seed list with neighbours that have improved reachability.
///
/// For each unprocessed neighbour `q`, the new reachability distance is
/// `max(core_dist_p, dist(p, q))`.  If this improves the current value, the
/// seed list is updated.
fn update_seeds<F: Float>(
    core_dist_p: F,
    neighbors: &[usize],
    neighbor_dists: &[F],
    processed: &[bool],
    reachability: &mut Array1<F>,
    seeds: &mut BinaryHeap<SeedEntry<F>>,
) {
    for (i, &q) in neighbors.iter().enumerate() {
        if processed[q] {
            continue;
        }
        let new_reach = if core_dist_p > neighbor_dists[i] {
            core_dist_p
        } else {
            neighbor_dists[i]
        };
        if new_reach < reachability[q] {
            reachability[q] = new_reach;
            seeds.push(SeedEntry {
                reach_dist: new_reach,
                idx: q,
            });
        }
    }
}

/// Xi-method cluster extraction from the reachability plot.
///
/// This is a simplified implementation:
/// - Find steep-down points (reachability drops by >= xi factor).
/// - Find steep-up points (reachability rises by >= xi factor).
/// - Match them to form clusters.
/// - Points not covered by any cluster are noise (-1).
fn xi_cluster_extraction<F: Float>(
    ordering: &[usize],
    reachability: &Array1<F>,
    xi: F,
) -> Array1<isize> {
    let n_ordered = ordering.len();
    let n_total = reachability.len();

    if n_ordered == 0 {
        return Array1::from_elem(n_total, -1isize);
    }

    // Build a reachability vector in ordering order, replacing infinity with
    // a sentinel large value for easier comparison.
    // We use the maximum finite reachability as our "infinity" sentinel.
    let max_finite = reachability
        .iter()
        .filter(|v| v.is_finite())
        .cloned()
        .fold(F::zero(), |acc, v| if v > acc { v } else { acc });

    let r_ord: Vec<F> = ordering
        .iter()
        .map(|&i| {
            let v = reachability[i];
            if v.is_finite() {
                v
            } else {
                max_finite + F::one()
            }
        })
        .collect();

    // Identify steep-down and steep-up positions.
    // A position i is steep-down if r_ord[i+1] <= (1 - xi) * r_ord[i].
    // A position i is steep-up   if r_ord[i]   <= (1 - xi) * r_ord[i-1].
    let one_minus_xi = F::one() - xi;

    let mut steep_down: Vec<usize> = Vec::new();
    let mut steep_up: Vec<usize> = Vec::new();

    for i in 0..(n_ordered.saturating_sub(1)) {
        if r_ord[i] == F::zero() {
            continue;
        }
        let ratio_next = r_ord[i + 1] / r_ord[i];
        if ratio_next <= one_minus_xi {
            steep_down.push(i);
        }
    }
    for i in 1..n_ordered {
        if r_ord[i - 1] == F::zero() {
            continue;
        }
        let ratio_prev = r_ord[i] / r_ord[i - 1];
        // Steep up: r[i] / r[i-1] >= 1/(1-xi)
        if ratio_prev >= F::one() / one_minus_xi {
            steep_up.push(i);
        }
    }

    // Build cluster intervals by matching steep-down starts with steep-up ends.
    let mut labels = Array1::from_elem(n_total, -1isize);
    let mut cluster_id: isize = 0;

    for &sd in &steep_down {
        // Look for the nearest steep-up point that comes after sd.
        if let Some(&su) = steep_up.iter().find(|&&su| su > sd) {
            // The cluster spans ordering positions [sd, su].
            let start = sd;
            let end = su;
            if end > start {
                for &pt in ordering[start..=end].iter() {
                    if labels[pt] == -1 {
                        labels[pt] = cluster_id;
                    }
                }
                cluster_id += 1;
            }
        }
    }

    labels
}

// ─────────────────────────────────────────────────────────────────────────────
// Fit implementation
// ─────────────────────────────────────────────────────────────────────────────

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for OPTICS<F> {
    type Fitted = FittedOPTICS<F>;
    type Error = FerroError;

    /// Fit the OPTICS model to the data.
    ///
    /// Computes the reachability ordering and distances for all training points.
    /// Cluster labels are extracted using the Xi method with the configured `xi`
    /// parameter.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `min_samples == 0`, `max_eps <= 0`,
    ///   or `xi` is outside `(0, 1)`.
    /// - [`FerroError::InsufficientSamples`] if the dataset is empty.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedOPTICS<F>, FerroError> {
        let n_samples = x.nrows();

        // Validate parameters.
        if self.min_samples == 0 {
            return Err(FerroError::InvalidParameter {
                name: "min_samples".into(),
                reason: "must be at least 1".into(),
            });
        }
        if self.max_eps <= F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "max_eps".into(),
                reason: "must be positive".into(),
            });
        }
        if self.xi <= F::zero() || self.xi >= F::one() {
            return Err(FerroError::InvalidParameter {
                name: "xi".into(),
                reason: "must be in (0, 1)".into(),
            });
        }

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "OPTICS requires at least 1 sample".into(),
            });
        }

        // Initialise state.
        let mut reachability = Array1::from_elem(n_samples, F::infinity());
        let mut core_distances = Array1::from_elem(n_samples, F::infinity());
        let mut processed = vec![false; n_samples];
        let mut ordering: Vec<usize> = Vec::with_capacity(n_samples);

        // Pre-compute core distances.
        for i in 0..n_samples {
            core_distances[i] = core_distance(x, i, self.max_eps, self.min_samples);
        }

        // Main OPTICS loop.
        for start in 0..n_samples {
            if processed[start] {
                continue;
            }

            processed[start] = true;
            ordering.push(start);

            if core_distances[start].is_infinite() {
                // Not a core point — just record it as noise candidate.
                continue;
            }

            // Priority queue (min-heap by reachability).
            let mut seeds: BinaryHeap<SeedEntry<F>> = BinaryHeap::new();

            let (nbrs, nbr_dists) = get_neighbors(x, start, self.max_eps);
            update_seeds(
                core_distances[start],
                &nbrs,
                &nbr_dists,
                &processed,
                &mut reachability,
                &mut seeds,
            );

            while let Some(entry) = seeds.pop() {
                let p = entry.idx;
                // Stale entry: a better reachability may have been inserted later.
                if processed[p] {
                    continue;
                }
                // Skip entries whose reachability is outdated.
                if entry.reach_dist > reachability[p] {
                    continue;
                }

                processed[p] = true;
                ordering.push(p);

                if core_distances[p].is_finite() {
                    let (p_nbrs, p_nbr_dists) = get_neighbors(x, p, self.max_eps);
                    update_seeds(
                        core_distances[p],
                        &p_nbrs,
                        &p_nbr_dists,
                        &processed,
                        &mut reachability,
                        &mut seeds,
                    );
                }
            }
        }

        // Extract cluster labels via the Xi method.
        let labels = xi_cluster_extraction(&ordering, &reachability, self.xi);

        Ok(FittedOPTICS {
            ordering_: ordering,
            reachability_: reachability,
            core_distances_: core_distances,
            labels_: labels,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Three tight 2-D clusters.
    fn three_blobs() -> Array2<f64> {
        Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1, 10.0, 0.0, 10.1, 0.0,
                10.0, 0.1,
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_ordering_covers_all_points() {
        let x = three_blobs();
        let fitted = OPTICS::<f64>::new(2).fit(&x, &()).unwrap();

        let mut sorted = fitted.ordering().to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..9).collect::<Vec<_>>());
    }

    #[test]
    fn test_reachability_length() {
        let x = three_blobs();
        let fitted = OPTICS::<f64>::new(2).fit(&x, &()).unwrap();
        assert_eq!(fitted.reachability().len(), 9);
    }

    #[test]
    fn test_core_distances_length() {
        let x = three_blobs();
        let fitted = OPTICS::<f64>::new(2).fit(&x, &()).unwrap();
        assert_eq!(fitted.core_distances().len(), 9);
    }

    #[test]
    fn test_labels_length() {
        let x = three_blobs();
        let fitted = OPTICS::<f64>::new(2).fit(&x, &()).unwrap();
        assert_eq!(fitted.labels().len(), 9);
    }

    #[test]
    fn test_core_points_have_finite_core_distance() {
        let x = three_blobs();
        // With min_samples=2 all tight-cluster points should be core points.
        let fitted = OPTICS::<f64>::new(2).fit(&x, &()).unwrap();
        // Points 0-8 each have at least one close neighbour.
        for i in 0..9 {
            // Core distance is finite because each point has neighbours.
            assert!(
                fitted.core_distances()[i].is_finite(),
                "expected finite core distance for point {i}"
            );
        }
    }

    #[test]
    fn test_isolated_point_infinite_core_distance() {
        // Add an isolated point far from the clusters.
        let mut data = three_blobs().into_raw_vec_and_offset().0;
        data.extend_from_slice(&[100.0, 100.0]);
        let x = Array2::from_shape_vec((10, 2), data).unwrap();

        // With max_eps=2.0, the isolated point has no neighbours, so its core
        // distance must be infinite regardless of min_samples.
        let fitted = OPTICS::<f64>::new(3)
            .with_max_eps(2.0)
            .fit(&x, &())
            .unwrap();
        assert!(
            fitted.core_distances()[9].is_infinite(),
            "isolated point should have infinite core distance"
        );
    }

    #[test]
    fn test_reachability_first_point_infinite() {
        let x = three_blobs();
        let fitted = OPTICS::<f64>::new(2).fit(&x, &()).unwrap();
        let first = fitted.ordering()[0];
        assert!(
            fitted.reachability()[first].is_infinite(),
            "first point in ordering should have infinite reachability"
        );
    }

    #[test]
    fn test_extract_clusters_valid_xi() {
        let x = three_blobs();
        let fitted = OPTICS::<f64>::new(2).fit(&x, &()).unwrap();
        let labels = fitted.extract_clusters(0.05).unwrap();
        assert_eq!(labels.len(), 9);
    }

    #[test]
    fn test_extract_clusters_invalid_xi_zero() {
        let x = three_blobs();
        let fitted = OPTICS::<f64>::new(2).fit(&x, &()).unwrap();
        assert!(fitted.extract_clusters(0.0).is_err());
    }

    #[test]
    fn test_extract_clusters_invalid_xi_one() {
        let x = three_blobs();
        let fitted = OPTICS::<f64>::new(2).fit(&x, &()).unwrap();
        assert!(fitted.extract_clusters(1.0).is_err());
    }

    #[test]
    fn test_invalid_min_samples_zero() {
        let x = three_blobs();
        let result = OPTICS::<f64>::new(0).fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_max_eps_zero() {
        let x = three_blobs();
        let result = OPTICS::<f64>::new(2).with_max_eps(0.0).fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_max_eps_negative() {
        let x = three_blobs();
        let result = OPTICS::<f64>::new(2).with_max_eps(-1.0).fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_xi_zero() {
        let x = three_blobs();
        let result = OPTICS::<f64>::new(2).with_xi(0.0).fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_xi_one() {
        let x = three_blobs();
        let result = OPTICS::<f64>::new(2).with_xi(1.0).fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_data_error() {
        let x = Array2::<f64>::zeros((0, 2));
        let result = OPTICS::<f64>::new(2).fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_single_sample() {
        let x = Array2::from_shape_vec((1, 2), vec![5.0, 5.0]).unwrap();
        let fitted = OPTICS::<f64>::new(1).fit(&x, &()).unwrap();
        assert_eq!(fitted.ordering().len(), 1);
        assert_eq!(fitted.ordering()[0], 0);
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

        let fitted = OPTICS::<f32>::new(2).fit(&x, &()).unwrap();
        assert_eq!(fitted.ordering().len(), 6);
    }

    #[test]
    fn test_n_clusters_non_negative() {
        let x = three_blobs();
        let fitted = OPTICS::<f64>::new(2).fit(&x, &()).unwrap();
        // n_clusters() just counts distinct non-noise labels.
        let _ = fitted.n_clusters(); // Should not panic.
    }

    #[test]
    fn test_ordering_unique_indices() {
        let x = three_blobs();
        let fitted = OPTICS::<f64>::new(2).fit(&x, &()).unwrap();
        let ordering = fitted.ordering();
        let mut seen = std::collections::HashSet::new();
        for &idx in ordering {
            assert!(seen.insert(idx), "duplicate index {idx} in ordering");
        }
    }

    #[test]
    fn test_with_max_eps_limits_reachability() {
        let x = three_blobs();
        let max_eps = 0.5;
        let fitted = OPTICS::<f64>::new(2)
            .with_max_eps(max_eps)
            .fit(&x, &())
            .unwrap();
        // All finite reachability values must be <= max_eps.
        for &r in fitted.reachability().iter() {
            if r.is_finite() {
                assert!(r <= max_eps + 1e-10);
            }
        }
    }
}
