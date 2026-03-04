//! KD-Tree for efficient nearest neighbor search.
//!
//! This module implements a KD-Tree (k-dimensional tree) that partitions
//! space by recursively splitting along the dimension with the greatest
//! spread, using the median value. Points are stored as indices into
//! the original dataset to avoid data duplication.
//!
//! # Complexity
//!
//! - Build: O(n log n)
//! - Query: O(log n) average case for low dimensions, degrades for d > 20
//!
//! # Examples
//!
//! ```
//! use ferrolearn_neighbors::kdtree::KdTree;
//! use ndarray::Array2;
//!
//! let data = Array2::from_shape_vec((4, 2), vec![
//!     0.0, 0.0,
//!     1.0, 0.0,
//!     0.0, 1.0,
//!     1.0, 1.0,
//! ]).unwrap();
//!
//! let tree = KdTree::build(&data);
//! // Find 2 nearest neighbors of (0.1, 0.1)
//! let query = vec![0.1_f64, 0.1];
//! let neighbors = tree.query(&data, &query, 2);
//! assert_eq!(neighbors[0].0, 0); // closest is (0,0)
//! ```

use ndarray::Array2;
use num_traits::Float;

/// A node in the KD-Tree.
///
/// Each node stores a point (by index) and optionally splits the space
/// along a dimension, with left and right children.
#[derive(Debug)]
struct KdNode {
    /// Index of the point stored at this node.
    index: usize,
    /// The dimension along which this node splits (if it has children).
    split_dim: usize,
    /// The value at which this node splits.
    split_val: f64,
    /// Left subtree (points with value <= split_val along split_dim).
    left: Option<Box<KdNode>>,
    /// Right subtree (points with value > split_val along split_dim).
    right: Option<Box<KdNode>>,
}

/// A KD-Tree spatial index for efficient nearest neighbor queries.
///
/// The tree stores indices into the original dataset rather than
/// copying point data. This means the original data must be available
/// at query time.
///
/// # Type Parameters
///
/// The tree is built from `Array2<F>` data but stores split values
/// as `f64` internally for simplicity. Queries convert to/from the
/// generic float type.
#[derive(Debug)]
pub struct KdTree {
    /// The root node of the tree, or `None` if the tree is empty.
    root: Option<Box<KdNode>>,
}

/// A bounded max-heap for maintaining the k nearest neighbors.
///
/// Stores `(distance, index)` pairs, with the largest distance at the top.
struct NeighborHeap {
    /// The maximum number of neighbors to keep.
    k: usize,
    /// Stored as `(distance, index)` pairs.
    items: Vec<(f64, usize)>,
}

impl NeighborHeap {
    /// Create a new empty heap with capacity `k`.
    fn new(k: usize) -> Self {
        Self {
            k,
            items: Vec::with_capacity(k + 1),
        }
    }

    /// Returns the current worst (largest) distance in the heap,
    /// or infinity if the heap is not yet full.
    fn worst_distance(&self) -> f64 {
        if self.items.len() < self.k {
            f64::INFINITY
        } else {
            self.items
                .iter()
                .fold(f64::NEG_INFINITY, |acc, &(d, _)| acc.max(d))
        }
    }

    /// Try to insert a neighbor. Only inserts if the distance is
    /// better than the current worst, or if the heap is not yet full.
    fn try_insert(&mut self, distance: f64, index: usize) {
        if self.items.len() < self.k {
            self.items.push((distance, index));
        } else {
            // Find the worst item.
            let worst_idx = self
                .items
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.0.partial_cmp(&b.1.0).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            if distance < self.items[worst_idx].0 {
                self.items[worst_idx] = (distance, index);
            }
        }
    }

    /// Drain the heap into a sorted vector of `(index, distance)` pairs,
    /// sorted by distance ascending.
    fn into_sorted(mut self) -> Vec<(usize, f64)> {
        self.items.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        self.items.into_iter().map(|(d, i)| (i, d)).collect()
    }
}

impl KdTree {
    /// Build a KD-Tree from a dataset.
    ///
    /// The tree is constructed by recursively splitting along the dimension
    /// with the greatest spread, using the median value as the split point.
    ///
    /// # Arguments
    ///
    /// - `data`: An `(n_samples, n_features)` array of points.
    ///
    /// # Returns
    ///
    /// A new `KdTree` that references points by index into `data`.
    pub fn build<F: Float + Send + Sync + 'static>(data: &Array2<F>) -> Self {
        let n_samples = data.nrows();
        if n_samples == 0 {
            return Self { root: None };
        }

        let n_features = data.ncols();
        let indices: Vec<usize> = (0..n_samples).collect();

        // Convert data to f64 for internal storage.
        let data_f64: Vec<Vec<f64>> = (0..n_samples)
            .map(|i| {
                (0..n_features)
                    .map(|j| data[[i, j]].to_f64().unwrap())
                    .collect()
            })
            .collect();

        let root = Self::build_recursive(&data_f64, &indices, n_features);
        Self {
            root: Some(Box::new(root)),
        }
    }

    /// Recursively build the tree.
    fn build_recursive(data: &[Vec<f64>], indices: &[usize], n_features: usize) -> KdNode {
        debug_assert!(!indices.is_empty());

        if indices.len() == 1 {
            return KdNode {
                index: indices[0],
                split_dim: 0,
                split_val: data[indices[0]][0],
                left: None,
                right: None,
            };
        }

        // Choose split dimension: the one with greatest spread.
        let split_dim = Self::choose_split_dimension(data, indices, n_features);

        // Sort indices by the chosen dimension.
        let mut sorted_indices = indices.to_vec();
        sorted_indices
            .sort_by(|&a, &b| data[a][split_dim].partial_cmp(&data[b][split_dim]).unwrap());

        // Median index.
        let median_pos = sorted_indices.len() / 2;
        let median_index = sorted_indices[median_pos];
        let split_val = data[median_index][split_dim];

        let left_indices = &sorted_indices[..median_pos];
        let right_indices = &sorted_indices[median_pos + 1..];

        let left = if left_indices.is_empty() {
            None
        } else {
            Some(Box::new(Self::build_recursive(
                data,
                left_indices,
                n_features,
            )))
        };

        let right = if right_indices.is_empty() {
            None
        } else {
            Some(Box::new(Self::build_recursive(
                data,
                right_indices,
                n_features,
            )))
        };

        KdNode {
            index: median_index,
            split_dim,
            split_val,
            left,
            right,
        }
    }

    /// Choose the dimension with the greatest spread (max - min).
    fn choose_split_dimension(data: &[Vec<f64>], indices: &[usize], n_features: usize) -> usize {
        (0..n_features)
            .max_by(|&dim_a, &dim_b| {
                let spread = |dim: usize| -> f64 {
                    let (min_val, max_val) = indices.iter().fold(
                        (f64::INFINITY, f64::NEG_INFINITY),
                        |(lo, hi), &idx| {
                            let v = data[idx][dim];
                            (lo.min(v), hi.max(v))
                        },
                    );
                    max_val - min_val
                };
                spread(dim_a)
                    .partial_cmp(&spread(dim_b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0)
    }

    /// Query the k nearest neighbors of a point.
    ///
    /// Returns a vector of `(index, distance)` pairs sorted by distance
    /// ascending. The distance is the Euclidean distance.
    ///
    /// # Arguments
    ///
    /// - `data`: The original dataset used to build the tree.
    /// - `query`: The query point as a slice of `f64` values.
    /// - `k`: The number of nearest neighbors to find.
    ///
    /// # Returns
    ///
    /// A vector of `(index, distance)` pairs, sorted by distance.
    pub fn query<F: Float + Send + Sync + 'static>(
        &self,
        data: &Array2<F>,
        query: &[f64],
        k: usize,
    ) -> Vec<(usize, f64)> {
        let n_features = data.ncols();

        // Convert data rows to f64 on the fly during search.
        let data_f64: Vec<Vec<f64>> = (0..data.nrows())
            .map(|i| {
                (0..n_features)
                    .map(|j| data[[i, j]].to_f64().unwrap())
                    .collect()
            })
            .collect();

        let mut heap = NeighborHeap::new(k);

        if let Some(root) = &self.root {
            Self::search_recursive(root, &data_f64, query, &mut heap);
        }

        heap.into_sorted()
    }

    /// Recursive search through the tree.
    fn search_recursive(node: &KdNode, data: &[Vec<f64>], query: &[f64], heap: &mut NeighborHeap) {
        // Check the current node's point.
        let dist = euclidean_distance_f64(&data[node.index], query);
        heap.try_insert(dist, node.index);

        // If this is a leaf (no children), we are done.
        if node.left.is_none() && node.right.is_none() {
            return;
        }

        // Determine which subtree to search first.
        let query_val = query[node.split_dim];
        let go_left = query_val <= node.split_val;

        let (first, second) = if go_left {
            (&node.left, &node.right)
        } else {
            (&node.right, &node.left)
        };

        // Always search the nearer subtree.
        if let Some(child) = first {
            Self::search_recursive(child, data, query, heap);
        }

        // Check if we need to search the farther subtree.
        let plane_dist = (query_val - node.split_val).abs();
        if plane_dist < heap.worst_distance() {
            if let Some(child) = second {
                Self::search_recursive(child, data, query, heap);
            }
        }
    }
}

/// Compute Euclidean distance between two points represented as `f64` slices.
fn euclidean_distance_f64(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| (ai - bi) * (ai - bi))
        .sum::<f64>()
        .sqrt()
}

/// Compute Euclidean distance between two points using a generic float type.
///
/// # Arguments
///
/// - `a`: First point as a slice.
/// - `b`: Second point as a slice.
///
/// # Returns
///
/// The Euclidean distance between `a` and `b`.
pub fn euclidean_distance<F: Float>(a: &[F], b: &[F]) -> F {
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| (ai - bi) * (ai - bi))
        .fold(F::zero(), |acc, v| acc + v)
        .sqrt()
}

/// Perform brute-force k-nearest-neighbor search.
///
/// Computes the Euclidean distance from the query point to every point
/// in the dataset and returns the `k` nearest points.
///
/// # Arguments
///
/// - `data`: The dataset as an `(n_samples, n_features)` array.
/// - `query`: The query point as a slice.
/// - `k`: The number of nearest neighbors to find.
///
/// # Returns
///
/// A vector of `(index, distance)` pairs sorted by distance ascending.
pub fn brute_force_knn<F: Float + Send + Sync + 'static>(
    data: &Array2<F>,
    query: &[F],
    k: usize,
) -> Vec<(usize, F)> {
    let n_samples = data.nrows();
    let n_features = data.ncols();

    // Compute all distances.
    let mut distances: Vec<(usize, F)> = (0..n_samples)
        .map(|i| {
            let point: Vec<F> = (0..n_features).map(|j| data[[i, j]]).collect();
            let dist = euclidean_distance(&point, query);
            (i, dist)
        })
        .collect();

    // Partial sort to get k smallest.
    let k_clamped = k.min(n_samples);
    if k_clamped == 0 {
        return Vec::new();
    }
    distances.select_nth_unstable_by(k_clamped - 1, |a, b| a.1.partial_cmp(&b.1).unwrap());
    distances.truncate(k_clamped);
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    distances
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_build_empty_tree() {
        let data = Array2::<f64>::zeros((0, 2));
        let tree = KdTree::build(&data);
        assert!(tree.root.is_none());
    }

    #[test]
    fn test_build_single_point() {
        let data = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
        let tree = KdTree::build(&data);
        assert!(tree.root.is_some());

        let neighbors = tree.query(&data, &[1.0, 2.0], 1);
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].0, 0);
        assert!((neighbors[0].1) < 1e-10);
    }

    #[test]
    fn test_query_simple() {
        let data =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();

        let tree = KdTree::build(&data);
        let neighbors = tree.query(&data, &[0.1, 0.1], 1);
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].0, 0); // (0,0) is closest to (0.1, 0.1)
    }

    #[test]
    fn test_query_k_neighbors() {
        let data = Array2::from_shape_vec(
            (5, 2),
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 10.0, 10.0],
        )
        .unwrap();

        let tree = KdTree::build(&data);
        let neighbors = tree.query(&data, &[0.5, 0.5], 4);
        assert_eq!(neighbors.len(), 4);

        // The 4 closest should be indices 0-3 (not 4, which is at (10,10)).
        let indices: Vec<usize> = neighbors.iter().map(|n| n.0).collect();
        assert!(!indices.contains(&4));
    }

    #[test]
    fn test_brute_force_simple() {
        let data =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();

        let neighbors = brute_force_knn(&data, &[0.1, 0.1], 2);
        assert_eq!(neighbors.len(), 2);
        assert_eq!(neighbors[0].0, 0); // (0,0) is closest
    }

    #[test]
    fn test_kdtree_matches_brute_force() {
        let data = Array2::from_shape_vec(
            (8, 2),
            vec![
                0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 2.0,
            ],
        )
        .unwrap();

        let tree = KdTree::build(&data);
        let query = [0.5, 0.5];

        for k in 1..=8 {
            let kd_result = tree.query(&data, &query, k);
            let bf_result = brute_force_knn(&data, &query, k);

            assert_eq!(kd_result.len(), bf_result.len(), "k={k}: length mismatch");

            // Compare distances (indices might differ for equidistant points).
            for (i, (kd, bf)) in kd_result.iter().zip(bf_result.iter()).enumerate() {
                assert!(
                    (kd.1 - bf.1).abs() < 1e-10,
                    "k={k}, neighbor {i}: kd dist={}, bf dist={}",
                    kd.1,
                    bf.1
                );
            }
        }
    }

    #[test]
    fn test_euclidean_distance() {
        let a = [0.0_f64, 0.0];
        let b = [3.0_f64, 4.0];
        assert!((euclidean_distance(&a, &b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_euclidean_distance_same_point() {
        let a = [1.0_f64, 2.0, 3.0];
        assert!(euclidean_distance(&a, &a) < 1e-15);
    }
}
