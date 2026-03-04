//! Ball tree for efficient nearest neighbor search in moderate-to-high dimensions.
//!
//! Unlike KD-Trees which partition along axis-aligned hyperplanes, ball trees
//! partition data into nested hyperspheres. This degrades more gracefully with
//! dimensionality, making ball trees effective for d > 15 where KD-Trees become
//! equivalent to brute force.
//!
//! # Complexity
//!
//! - Build: O(n log n)
//! - Query: O(n^(1-1/d) log n) amortized, much better than O(n) brute force
//!   for moderate dimensions
//!
//! # Examples
//!
//! ```
//! use ferrolearn_neighbors::balltree::BallTree;
//! use ndarray::Array2;
//!
//! let data = Array2::from_shape_vec((4, 2), vec![
//!     0.0, 0.0,
//!     1.0, 0.0,
//!     0.0, 1.0,
//!     1.0, 1.0,
//! ]).unwrap();
//!
//! let tree = BallTree::build(&data);
//! let neighbors = tree.query(&data, &[0.1, 0.1], 2);
//! assert_eq!(neighbors[0].0, 0); // closest is (0,0)
//! ```

use ndarray::Array2;
use num_traits::Float;

/// A node in the ball tree.
///
/// Each leaf node stores a range of point indices. Internal nodes store a
/// pivot point (the centroid of their subset) and a bounding radius.
#[derive(Debug)]
struct BallNode {
    /// Index of the centroid point (closest to the actual centroid).
    pivot: usize,
    /// Bounding radius: max distance from pivot to any point in this node.
    radius: f64,
    /// Indices into the original data (only populated for leaf nodes).
    /// For internal nodes, this is empty — children contain the indices.
    start: usize,
    /// End index (exclusive) into the index array.
    end: usize,
    /// Left child (if internal node).
    left: Option<Box<BallNode>>,
    /// Right child (if internal node).
    right: Option<Box<BallNode>>,
}

/// A ball tree spatial index for nearest neighbor queries.
///
/// Stores data as flattened f64 for cache-friendly access. Point indices
/// are permuted during construction so that each node's points are
/// contiguous in memory.
#[derive(Debug)]
pub struct BallTree {
    /// Root of the tree.
    root: Option<Box<BallNode>>,
    /// Flattened data: `data[i * n_features + j]` is feature j of point i.
    data: Vec<f64>,
    /// Number of features per point.
    n_features: usize,
    /// Permuted index array — maps internal indices to original dataset indices.
    indices: Vec<usize>,
}

/// Maximum number of points in a leaf node before splitting.
const LEAF_SIZE: usize = 40;

/// A bounded max-heap for tracking the k nearest neighbors.
struct NeighborHeap {
    k: usize,
    items: Vec<(f64, usize)>, // (distance, original_index)
}

impl NeighborHeap {
    fn new(k: usize) -> Self {
        Self {
            k,
            items: Vec::with_capacity(k + 1),
        }
    }

    /// Worst (largest) distance currently tracked, or infinity if not full.
    fn worst_distance(&self) -> f64 {
        if self.items.len() < self.k {
            f64::INFINITY
        } else {
            self.items
                .iter()
                .fold(f64::NEG_INFINITY, |acc, &(d, _)| acc.max(d))
        }
    }

    /// Try to insert a neighbor. Only inserts if better than worst or heap not full.
    fn try_insert(&mut self, distance: f64, index: usize) {
        if self.items.len() < self.k {
            self.items.push((distance, index));
        } else {
            let worst_idx = self
                .items
                .iter()
                .enumerate()
                .max_by(|a, b| a.1 .0.partial_cmp(&b.1 .0).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            if distance < self.items[worst_idx].0 {
                self.items[worst_idx] = (distance, index);
            }
        }
    }

    /// Drain into sorted `(original_index, distance)` pairs.
    fn into_sorted(mut self) -> Vec<(usize, f64)> {
        self.items.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        self.items.into_iter().map(|(d, i)| (i, d)).collect()
    }
}

impl BallTree {
    /// Build a ball tree from a dataset.
    ///
    /// Converts data to f64 internally and constructs a hierarchical
    /// bounding-ball partition.
    pub fn build<F: Float + Send + Sync + 'static>(data: &Array2<F>) -> Self {
        let n_samples = data.nrows();
        let n_features = data.ncols();

        if n_samples == 0 {
            return Self {
                root: None,
                data: Vec::new(),
                n_features,
                indices: Vec::new(),
            };
        }

        // Flatten data to f64 for cache-friendly access.
        let flat_data: Vec<f64> = (0..n_samples)
            .flat_map(|i| (0..n_features).map(move |j| data[[i, j]].to_f64().unwrap()))
            .collect();

        let mut indices: Vec<usize> = (0..n_samples).collect();

        let root = Self::build_recursive(&flat_data, &mut indices, 0, n_samples, n_features);

        Self {
            root: Some(Box::new(root)),
            data: flat_data,
            n_features,
            indices,
        }
    }

    /// Recursively build the tree over indices[start..end].
    fn build_recursive(
        data: &[f64],
        indices: &mut [usize],
        start: usize,
        end: usize,
        n_features: usize,
    ) -> BallNode {
        let count = end - start;

        // Find centroid of this subset.
        let centroid = compute_centroid(data, &indices[start..end], n_features);

        // Find the point closest to the centroid — that's our pivot.
        let pivot_pos = find_closest_to_centroid(data, &indices[start..end], &centroid, n_features);
        let pivot = indices[start + pivot_pos];

        // Compute bounding radius: max distance from pivot to any point.
        let radius = compute_radius(data, &indices[start..end], pivot, n_features);

        // If small enough, make a leaf.
        if count <= LEAF_SIZE {
            return BallNode {
                pivot,
                radius,
                start,
                end,
                left: None,
                right: None,
            };
        }

        // Split: find the dimension of greatest spread.
        let split_dim = dimension_of_greatest_spread(data, &indices[start..end], n_features);

        // Partition around the median along split_dim.
        let mid = start + count / 2;
        partition_by_dimension(data, &mut indices[start..end], split_dim, n_features);
        // Now indices[start..mid] have smaller values along split_dim,
        // indices[mid..end] have larger values.

        let left = Self::build_recursive(data, indices, start, mid, n_features);
        let right = Self::build_recursive(data, indices, mid, end, n_features);

        BallNode {
            pivot,
            radius,
            start,
            end,
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
        }
    }

    /// Query the k nearest neighbors of a point.
    ///
    /// Returns `(original_index, distance)` pairs sorted by distance ascending.
    pub fn query<F: Float + Send + Sync + 'static>(
        &self,
        _data: &Array2<F>,
        query: &[f64],
        k: usize,
    ) -> Vec<(usize, f64)> {
        let mut heap = NeighborHeap::new(k);

        if let Some(root) = &self.root {
            self.search_recursive(root, query, &mut heap);
        }

        heap.into_sorted()
    }

    /// Recursive search with ball pruning.
    fn search_recursive(&self, node: &BallNode, query: &[f64], heap: &mut NeighborHeap) {
        let nf = self.n_features;

        // Distance from query to this node's pivot.
        let pivot_data = &self.data[node.pivot * nf..(node.pivot + 1) * nf];
        let dist_to_pivot = euclidean_dist(query, pivot_data);

        // Ball pruning: if the closest possible point in this ball is
        // farther than our current worst, skip entirely.
        let min_possible = (dist_to_pivot - node.radius).max(0.0);
        if min_possible >= heap.worst_distance() {
            return;
        }

        // If leaf, check all points.
        if node.left.is_none() && node.right.is_none() {
            for &idx in &self.indices[node.start..node.end] {
                let point = &self.data[idx * nf..(idx + 1) * nf];
                let d = euclidean_dist(query, point);
                heap.try_insert(d, idx);
            }
            return;
        }

        // Check the pivot point.
        heap.try_insert(dist_to_pivot, node.pivot);

        // Determine which child is closer and search it first.
        let (closer, farther) = match (&node.left, &node.right) {
            (Some(l), Some(r)) => {
                let l_pivot = &self.data[l.pivot * nf..(l.pivot + 1) * nf];
                let r_pivot = &self.data[r.pivot * nf..(r.pivot + 1) * nf];
                let dl = euclidean_dist(query, l_pivot);
                let dr = euclidean_dist(query, r_pivot);
                if dl <= dr {
                    (l.as_ref(), r.as_ref())
                } else {
                    (r.as_ref(), l.as_ref())
                }
            }
            (Some(l), None) => {
                self.search_recursive(l, query, heap);
                return;
            }
            (None, Some(r)) => {
                self.search_recursive(r, query, heap);
                return;
            }
            (None, None) => unreachable!(),
        };

        self.search_recursive(closer, query, heap);
        self.search_recursive(farther, query, heap);
    }
}

/// Euclidean distance between two f64 slices.
#[inline]
fn euclidean_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| (ai - bi) * (ai - bi))
        .sum::<f64>()
        .sqrt()
}

/// Compute the centroid of a set of points.
fn compute_centroid(data: &[f64], indices: &[usize], n_features: usize) -> Vec<f64> {
    let n = indices.len() as f64;
    let mut centroid = vec![0.0; n_features];
    for &idx in indices {
        let base = idx * n_features;
        for j in 0..n_features {
            centroid[j] += data[base + j];
        }
    }
    for v in &mut centroid {
        *v /= n;
    }
    centroid
}

/// Find the index (position within the slice) of the point closest to the centroid.
fn find_closest_to_centroid(
    data: &[f64],
    indices: &[usize],
    centroid: &[f64],
    n_features: usize,
) -> usize {
    let mut best_pos = 0;
    let mut best_dist = f64::INFINITY;
    for (pos, &idx) in indices.iter().enumerate() {
        let point = &data[idx * n_features..(idx + 1) * n_features];
        let d = euclidean_dist(point, centroid);
        if d < best_dist {
            best_dist = d;
            best_pos = pos;
        }
    }
    best_pos
}

/// Compute the max distance from a pivot to any point in the set.
fn compute_radius(data: &[f64], indices: &[usize], pivot: usize, n_features: usize) -> f64 {
    let pivot_data = &data[pivot * n_features..(pivot + 1) * n_features];
    let mut max_dist = 0.0_f64;
    for &idx in indices {
        let point = &data[idx * n_features..(idx + 1) * n_features];
        let d = euclidean_dist(pivot_data, point);
        if d > max_dist {
            max_dist = d;
        }
    }
    max_dist
}

/// Find the dimension with the greatest spread among the given points.
fn dimension_of_greatest_spread(data: &[f64], indices: &[usize], n_features: usize) -> usize {
    let mut best_dim = 0;
    let mut best_spread = f64::NEG_INFINITY;
    for dim in 0..n_features {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for &idx in indices {
            let v = data[idx * n_features + dim];
            lo = lo.min(v);
            hi = hi.max(v);
        }
        let spread = hi - lo;
        if spread > best_spread {
            best_spread = spread;
            best_dim = dim;
        }
    }
    best_dim
}

/// Partition indices around the median value along the given dimension.
/// After this call, indices[..mid] have smaller values and indices[mid..]
/// have larger values along `dim`.
fn partition_by_dimension(data: &[f64], indices: &mut [usize], dim: usize, n_features: usize) {
    let mid = indices.len() / 2;
    indices.select_nth_unstable_by(mid, |&a, &b| {
        let va = data[a * n_features + dim];
        let vb = data[b * n_features + dim];
        va.partial_cmp(&vb).unwrap()
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kdtree;
    use ndarray::Array2;

    #[test]
    fn test_build_empty() {
        let data = Array2::<f64>::zeros((0, 2));
        let tree = BallTree::build(&data);
        assert!(tree.root.is_none());
    }

    #[test]
    fn test_build_single_point() {
        let data = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
        let tree = BallTree::build(&data);
        assert!(tree.root.is_some());

        let neighbors = tree.query(&data, &[1.0, 2.0], 1);
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].0, 0);
        assert!(neighbors[0].1 < 1e-10);
    }

    #[test]
    fn test_query_simple() {
        let data =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();

        let tree = BallTree::build(&data);
        let neighbors = tree.query(&data, &[0.1, 0.1], 1);
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].0, 0);
    }

    #[test]
    fn test_query_k_neighbors() {
        let data = Array2::from_shape_vec(
            (5, 2),
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 10.0, 10.0],
        )
        .unwrap();

        let tree = BallTree::build(&data);
        let neighbors = tree.query(&data, &[0.5, 0.5], 4);
        assert_eq!(neighbors.len(), 4);

        // The 4 closest should be indices 0-3 (not 4, at (10,10)).
        let indices: Vec<usize> = neighbors.iter().map(|n| n.0).collect();
        assert!(!indices.contains(&4));
    }

    #[test]
    fn test_balltree_matches_brute_force() {
        let data = Array2::from_shape_vec(
            (8, 2),
            vec![
                0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 2.0,
            ],
        )
        .unwrap();

        let tree = BallTree::build(&data);
        let query = [0.5, 0.5];

        for k in 1..=8 {
            let bt_result = tree.query(&data, &query, k);
            let bf_result = kdtree::brute_force_knn(&data, &query, k);

            assert_eq!(bt_result.len(), bf_result.len(), "k={k}: length mismatch");

            for (i, (bt, bf)) in bt_result.iter().zip(bf_result.iter()).enumerate() {
                assert!(
                    (bt.1 - bf.1).abs() < 1e-10,
                    "k={k}, neighbor {i}: bt dist={}, bf dist={}",
                    bt.1,
                    bf.1
                );
            }
        }
    }

    #[test]
    fn test_high_dimensional() {
        // 50 points in 100 dimensions — the main use case.
        let n = 50;
        let d = 100;
        let flat: Vec<f64> = (0..n * d).map(|i| (i as f64) * 0.01).collect();
        let data = Array2::from_shape_vec((n, d), flat).unwrap();

        let tree = BallTree::build(&data);
        let query: Vec<f64> = vec![0.5; d];

        let bt_result = tree.query(&data, &query, 5);
        let bf_result = kdtree::brute_force_knn(&data, &query, 5);

        assert_eq!(bt_result.len(), bf_result.len());
        for (bt, bf) in bt_result.iter().zip(bf_result.iter()) {
            assert!(
                (bt.1 - bf.1).abs() < 1e-10,
                "bt dist={}, bf dist={}",
                bt.1,
                bf.1
            );
        }
    }
}
