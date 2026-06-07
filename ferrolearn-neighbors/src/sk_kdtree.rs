//! Bit-exact port of scikit-learn's `KDTree` build + single-tree depth-first
//! k-NN query, for the dense f64 euclidean (`minkowski`, `p=2`) default path.
//!
//! sklearn's default `KNeighborsClassifier`/`KNeighborsRegressor`
//! (`metric='minkowski'`, `p=2`, `leaf_size=30`) routes a low-dimensional,
//! small-`k` query through `kd_tree` (`sklearn/neighbors/_base.py:607-640`
//! auto-selection), which calls `KDTree.query` — NOT the brute `ArgKmin`
//! backend. The KDTree's leaf push order on an exact distance tie depends on
//! the exact index layout produced during the build, so to reproduce sklearn's
//! `kd_tree` k-NN SET and ORDER bit-for-bit we replicate:
//!
//! - `_recursive_build` / `init_node` / `find_node_split_dim` /
//!   `partition_node_indices` (`sklearn/neighbors/_binary_tree.pxi.tp:1037-1087`,
//!   `sklearn/neighbors/_kd_tree.pyx.tp:73-120`), with the partition delegated
//!   to the bit-exact `std::nth_element` port in [`crate::introselect`]
//!   (`sklearn/neighbors/_partition_nodes.pyx`).
//! - `_query_single_depthfirst` + `min_rdist` (squared euclidean lower bound)
//!   (`_binary_tree.pxi.tp:1607-1659`, `_kd_tree.pyx.tp:123-147`).
//!
//! The query reuses the same `NeighborsHeap` kernel sklearn uses
//! (`heap_push` + `simultaneous_sort` in [`crate::knn`]), pushing SQUARED
//! euclidean distances; ties at the heap boundary are rejected, exactly as in
//! `BinaryTree.query`. Distances are returned as non-squared euclidean
//! (`rdist_to_dist`) so the caller's distance weighting is unchanged.
//!
//! leaf_size is fixed at 30, the value `KNeighbors*` passes (sklearn
//! `leaf_size=30` default, NOT the `BinaryTree` default 40).
//!
//! No panic path: all indexing is bounds-checked by construction; the public
//! [`SkKdTree::query`] takes `k <= n_samples` (the caller guards `k > n`
//! upstream, mirroring sklearn's query-time `ValueError`).

use ndarray::Array2;

use crate::introselect::nth_element;
use crate::knn::{heap_push, simultaneous_sort};

/// sklearn `KNeighbors*` `leaf_size` default (`_classification.py:199`).
const LEAF_SIZE: usize = 30;

/// Per-node metadata, mirroring sklearn's `NodeData_t`
/// (`_binary_tree.pxi.tp`). `radius` is stored for fidelity with the build but
/// is not consulted by single-tree k-NN query pruning.
#[derive(Clone, Copy, Debug)]
struct NodeData {
    idx_start: usize,
    idx_end: usize,
    is_leaf: bool,
    #[allow(
        dead_code,
        reason = "stored for build fidelity; query prunes via min_rdist not radius"
    )]
    radius: f64,
}

impl Default for NodeData {
    fn default() -> Self {
        NodeData {
            idx_start: 0,
            idx_end: 0,
            is_leaf: false,
            radius: 0.0,
        }
    }
}

/// A bit-exact sklearn KDTree over a dense f64 dataset, for euclidean (`p=2`)
/// single-tree k-NN queries.
#[derive(Debug)]
pub struct SkKdTree {
    /// Row-major copy of the training data, `n_samples * n_features`.
    data: Vec<f64>,
    n_samples: usize,
    n_features: usize,
    /// Permutation of `0..n_samples` partitioned per node (sklearn `idx_array`).
    idx_array: Vec<usize>,
    /// Per-node bounds: `lower[node][feat]` / `upper[node][feat]` flattened
    /// row-major as `node * n_features + feat`.
    lower_bounds: Vec<f64>,
    upper_bounds: Vec<f64>,
    node_data: Vec<NodeData>,
    n_nodes: usize,
}

impl SkKdTree {
    /// Build a KDTree from a dense f64 array, mirroring sklearn's
    /// `BinaryTree.__init__` + `_recursive_build` with `leaf_size = 30`.
    #[must_use]
    pub fn build(data: &Array2<f64>) -> Self {
        let n_samples = data.nrows();
        let n_features = data.ncols();

        // Flatten to row-major (sklearn stores `self.data` C-contiguous).
        let mut flat = vec![0.0f64; n_samples * n_features];
        for i in 0..n_samples {
            for j in 0..n_features {
                flat[i * n_features + j] = data[[i, j]];
            }
        }

        // n_levels = int(log2(fmax(1, (n_samples - 1) / leaf_size)) + 1)
        // n_nodes  = 2**n_levels - 1
        // (`_binary_tree.pxi.tp:879-881`). Use float division as upstream does.
        let ratio = if n_samples == 0 {
            0.0
        } else {
            ((n_samples - 1) as f64) / (LEAF_SIZE as f64)
        };
        let n_levels = (f64::log2(ratio.max(1.0)) + 1.0) as usize;
        let n_nodes = (1usize << n_levels) - 1;

        let idx_array: Vec<usize> = (0..n_samples).collect();
        let lower_bounds = vec![0.0f64; n_nodes * n_features];
        let upper_bounds = vec![0.0f64; n_nodes * n_features];
        let node_data = vec![NodeData::default(); n_nodes];

        let mut tree = SkKdTree {
            data: flat,
            n_samples,
            n_features,
            idx_array,
            lower_bounds,
            upper_bounds,
            node_data,
            n_nodes,
        };

        if n_samples > 0 {
            tree.recursive_build(0, 0, n_samples);
        }
        tree
    }

    /// sklearn `init_node` (`_kd_tree.pyx.tp:73-120`) for `p=2`: compute the
    /// per-feature bounding box of the node's points and its circumscribing
    /// radius.
    fn init_node(&mut self, i_node: usize, idx_start: usize, idx_end: usize) {
        let nf = self.n_features;
        for j in 0..nf {
            self.lower_bounds[i_node * nf + j] = f64::INFINITY;
            self.upper_bounds[i_node * nf + j] = f64::NEG_INFINITY;
        }
        for i in idx_start..idx_end {
            let row = self.idx_array[i] * nf;
            for j in 0..nf {
                let v = self.data[row + j];
                let lo = &mut self.lower_bounds[i_node * nf + j];
                *lo = lo.min(v);
                let hi = &mut self.upper_bounds[i_node * nf + j];
                *hi = hi.max(v);
            }
        }
        // p=2: rad += (0.5 * |upper - lower|)^2 ; radius = rad^(1/2).
        let mut rad = 0.0f64;
        for j in 0..nf {
            let half = 0.5
                * (self.upper_bounds[i_node * nf + j] - self.lower_bounds[i_node * nf + j]).abs();
            rad += half * half;
        }
        self.node_data[i_node] = NodeData {
            idx_start,
            idx_end,
            is_leaf: self.node_data[i_node].is_leaf,
            radius: rad.sqrt(),
        };
    }

    /// sklearn `find_node_split_dim` (`_binary_tree.pxi.tp:601-648`): the
    /// feature with the largest spread, FIRST on ties (strict `>`,
    /// `max_spread` initialised to 0).
    fn find_node_split_dim(&self, idx_start: usize, n_points: usize) -> usize {
        let nf = self.n_features;
        let mut j_max = 0usize;
        let mut max_spread = 0.0f64;
        for j in 0..nf {
            let first_row = self.idx_array[idx_start] * nf;
            let mut max_val = self.data[first_row + j];
            let mut min_val = max_val;
            for i in 1..n_points {
                let row = self.idx_array[idx_start + i] * nf;
                let v = self.data[row + j];
                max_val = max_val.max(v);
                min_val = min_val.min(v);
            }
            let spread = max_val - min_val;
            if spread > max_spread {
                max_spread = spread;
                j_max = j;
            }
        }
        j_max
    }

    /// sklearn `partition_node_indices` (`_partition_nodes.pyx`): partition the
    /// node's slice of `idx_array` about `split_index = n_mid` on `split_dim`,
    /// via the bit-exact `std::nth_element` port. Comparator is the upstream
    /// `IndexComparator`: tie on the feature value breaks to the lower index.
    fn partition_node_indices(
        &mut self,
        idx_start: usize,
        n_points: usize,
        split_dim: usize,
        n_mid: usize,
    ) {
        let nf = self.n_features;
        // Snapshot the data column reads as a closure over the immutable data;
        // borrow the mutable slice of idx_array separately.
        let data = &self.data;
        let cmp = |a: usize, b: usize| -> bool {
            let av = data[a * nf + split_dim];
            let bv = data[b * nf + split_dim];
            if av == bv { a < b } else { av < bv }
        };
        let slice = &mut self.idx_array[idx_start..idx_start + n_points];
        nth_element(slice, n_mid, &cmp);
    }

    /// sklearn `_recursive_build` (`_binary_tree.pxi.tp:1037-1087`).
    #[allow(
        clippy::if_same_then_else,
        reason = "faithful 1:1 port of sklearn _recursive_build's two distinct \
                  leaf conditions (`2*i_node+1 >= n_nodes` capacity exhausted vs \
                  `idx_end-idx_start < 2` too-few-points), _binary_tree.pxi.tp:1059-1075"
    )]
    fn recursive_build(&mut self, i_node: usize, idx_start: usize, idx_end: usize) {
        let n_points = idx_end - idx_start;
        let n_mid = n_points / 2;

        self.init_node(i_node, idx_start, idx_end);

        if 2 * i_node + 1 >= self.n_nodes {
            self.node_data[i_node].is_leaf = true;
        } else if idx_end - idx_start < 2 {
            self.node_data[i_node].is_leaf = true;
        } else {
            self.node_data[i_node].is_leaf = false;
            let i_max = self.find_node_split_dim(idx_start, n_points);
            self.partition_node_indices(idx_start, n_points, i_max, n_mid);
            self.recursive_build(2 * i_node + 1, idx_start, idx_start + n_mid);
            self.recursive_build(2 * i_node + 2, idx_start + n_mid, idx_end);
        }
    }

    /// sklearn `min_rdist` for `p=2` (`_kd_tree.pyx.tp:123-147`): the squared
    /// euclidean lower-bound distance from `pt` to node `i_node`'s box.
    fn min_rdist(&self, i_node: usize, pt: &[f64]) -> f64 {
        let nf = self.n_features;
        let lo = &self.lower_bounds[i_node * nf..i_node * nf + nf];
        let hi = &self.upper_bounds[i_node * nf..i_node * nf + nf];
        let mut rdist = 0.0f64;
        for ((&p, &l), &u) in pt.iter().zip(lo).zip(hi) {
            let d_lo = l - p;
            let d_hi = p - u;
            // 0.5 * ((d_lo + |d_lo|) + (d_hi + |d_hi|)) = max(d_lo,0)+max(d_hi,0).
            let d = (d_lo + d_lo.abs()) + (d_hi + d_hi.abs());
            let half = 0.5 * d;
            rdist += half * half;
        }
        rdist
    }

    /// Squared euclidean distance between `pt` and training row `idx`.
    fn rdist(&self, pt: &[f64], idx: usize) -> f64 {
        let nf = self.n_features;
        let row = &self.data[idx * nf..idx * nf + nf];
        let mut acc = 0.0f64;
        for (&r, &p) in row.iter().zip(pt) {
            let d = r - p;
            acc += d * d;
        }
        acc
    }

    /// sklearn `_query_single_depthfirst` (`_binary_tree.pxi.tp:1607-1659`):
    /// recursive single-tree depth-first k-NN, pushing squared distances onto
    /// the `NeighborsHeap` (`hv`/`hi`), nearest child first.
    #[allow(
        clippy::too_many_arguments,
        reason = "faithful 1:1 port of sklearn _query_single_depthfirst signature \
                  (i_node, pt, heap, reduced_dist_LB) plus the heap held as SoA"
    )]
    fn query_single_depthfirst(
        &self,
        i_node: usize,
        pt: &[f64],
        hv: &mut [f64],
        hi: &mut [usize],
        k: usize,
        reduced_dist_lb: f64,
    ) {
        let node = self.node_data[i_node];

        // Case 1: query point outside node radius -> trim.
        if reduced_dist_lb > hv[0] {
            return;
        }

        // Case 2: leaf -> update the heap with every point in the leaf.
        if node.is_leaf {
            for i in node.idx_start..node.idx_end {
                let idx = self.idx_array[i];
                let d = self.rdist(pt, idx);
                heap_push(hv, hi, k, d, idx);
            }
            return;
        }

        // Case 3: split -> recurse into the closer child first.
        let i1 = 2 * i_node + 1;
        let i2 = i1 + 1;
        let lb1 = self.min_rdist(i1, pt);
        let lb2 = self.min_rdist(i2, pt);
        if lb1 <= lb2 {
            self.query_single_depthfirst(i1, pt, hv, hi, k, lb1);
            self.query_single_depthfirst(i2, pt, hv, hi, k, lb2);
        } else {
            self.query_single_depthfirst(i2, pt, hv, hi, k, lb2);
            self.query_single_depthfirst(i1, pt, hv, hi, k, lb1);
        }
    }

    /// Query the `k` nearest neighbors of `pt`, returning `(indices,
    /// euclidean_distances)` nearest-first — sklearn `KDTree.query` for one
    /// row with `sort_results=True`.
    ///
    /// `k` must satisfy `1 <= k <= n_samples` (the caller guards `k > n`).
    #[must_use]
    pub fn query(&self, pt: &[f64], k: usize) -> (Vec<usize>, Vec<f64>) {
        let mut hv = vec![f64::INFINITY; k];
        let mut hi = vec![0usize; k];

        if self.n_samples == 0 || k == 0 {
            return (hi, hv);
        }

        let reduced_dist_lb = self.min_rdist(0, pt);
        self.query_single_depthfirst(0, pt, &mut hv, &mut hi, k, reduced_dist_lb);

        // sort ascending (sklearn `simultaneous_sort`), then rdist -> dist.
        simultaneous_sort(&mut hv, &mut hi, k);
        for v in hv.iter_mut() {
            *v = v.max(0.0).sqrt();
        }
        (hi, hv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// The KDTree must reproduce sklearn `KDTree.query` index ORDER on the
    /// non-tie distinct-distance fixture (every algorithm agrees here).
    /// Oracle: live sklearn 1.5.2 (system python3).
    #[test]
    fn query_distinct_distances() {
        let x = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]];
        let tree = SkKdTree::build(&x);
        let (idx, _d) = tree.query(&[0.0, 0.0], 3);
        // sklearn KDTree(x, leaf_size=30).query([[0,0]], 3) -> [[0,1,2]]
        assert_eq!(idx, vec![0, 1, 2]);
    }

    /// Build sanity: idx_array is a permutation of 0..n.
    #[test]
    fn build_idx_array_is_permutation() {
        let x = array![[3.0, 1.0], [1.0, 4.0], [2.0, 2.0], [5.0, 0.0], [0.0, 3.0],];
        let tree = SkKdTree::build(&x);
        let mut seen = tree.idx_array.clone();
        seen.sort_unstable();
        assert_eq!(seen, vec![0, 1, 2, 3, 4]);
    }
}
