//! Adversarial divergence pins for `ferrolearn-neighbors/src/kdtree.rs`
//! (`pub struct KdTree`, `KdTree::build`, `KdTree::query`) against the live
//! scikit-learn 1.5.2 oracle (`from sklearn.neighbors import KDTree`).
//!
//! Every expected value below is captured from a live `python3 -c "..."` run of
//! sklearn 1.5.2 (R-CHAR-3 — never literal-copied from the ferrolearn side).
//! The exact oracle call and its `(d, i)` output is quoted above each assertion.
//!
//! Module REQ table: REQ-1 (query value) SHIPPED; REQ-2 (k>n error) NOT-STARTED
//! (#831). Design doc: `.design/neighbors/kdtree.md`.
//! Upstream:
//!   * `sklearn/neighbors/_binary_tree.pxi.tp:1140-1142` — `query` raises
//!     `ValueError("k must be less than or equal to the number of training
//!     points")` when `self.data.shape[0] < k`.
//!   * `sklearn/neighbors/_binary_tree.pxi.tp:855-856` — `__init__` raises
//!     `ValueError("X is an empty array")` on empty `X` (in practice
//!     `check_array` at :854 raises first: "Found array with 0 sample(s) ...").
//!   * `sklearn/neighbors/_binary_tree.pxi.tp:1191` — `sort_results=True`
//!     returns each row's `k` neighbors sorted nearest-distance-first.
//!
//! ferrolearn API under test (the signature the fixer consumes):
//!   * `KdTree::build<F: Float + ..>(data: &Array2<F>) -> KdTree`
//!   * `KdTree::query<F: Float + ..>(&self, data: &Array2<F>, query: &[f64],
//!     k: usize) -> Vec<(usize, f64)>`   — flat `(index, true-euclidean-dist)`
//!     pairs, sorted ascending. NOTE: returns a `Vec`, NOT a `Result`, so the
//!     `k > n_samples` divergence is "silently returns `min(k, n)` neighbors"
//!     rather than raising — the RED pin asserts the sklearn contract that no
//!     such truncated answer is produced.
//!
//! RED pins (must FAIL against current kdtree.rs — deterministic divergences):
//!   * `divergence_query_k_gt_nsamples_must_not_silently_truncate` — blocker for
//!     the `k > n_samples` error contract (`:1140-1142`).
//!
//! GREEN guards (must PASS — REQ-2 shipped single-row k-NN value contract):
//!   * `green_query_k2_distinct_distances_and_indices`
//!   * `green_query_k1_nearest_neighbor`
//!   * `green_query_k4_tie_set_and_sorted_distances`
//!
//! Empty-input error (`:855-856`) is NOT pinned here: `KdTree::build` returns
//! `KdTree` (not `Result`) and the empty/None state is a private field, so the
//! sklearn "raise on empty X" contract is not expressible through the current
//! public API. Tracked as a NOT-STARTED REQ-7 blocker; no forced test (per the
//! critic brief: SKIP and note when the API can't express it).
//!
//! NOTE on tie-free query points: `(0.1,0.1)` is NOT tie-free against this
//! dataset — points (1,0) and (0,1) are both at distance sqrt(0.82) from it, so
//! its k>=2 index order is a TIE (traversal-dependent), not a distinct order.
//! The distinct-distance guards below therefore use `(0.2,0.1)`, whose
//! distances to all five points are strictly increasing and distinct
//! (0.2236 < 0.8062 < 0.9219 < 1.2041 < 13.93), giving a deterministic index
//! order `[0,1,2,...]` that is a real value-parity assertion (not a tie).

use ferrolearn_neighbors::kdtree::KdTree;
use ndarray::{Array2, array};

/// The fixed dataset used across the oracle calls below.
/// `X = [[0,0],[1,0],[0,1],[1,1],[10,10]]` (n_samples = 5).
fn five_points() -> Array2<f64> {
    array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [10.0, 10.0]]
}

// sqrt(0.5) — the common tie distance from (0.5,0.5) to all four unit-square
// corners. python3 -c "import math;print(repr(math.sqrt(0.5)))"
// -> 0.7071067811865476
const SQRT_HALF: f64 = std::f64::consts::FRAC_1_SQRT_2;

// ===========================================================================
// GREEN 1 — REQ-2 shipped: k=2 + k=3 distinct-distance distances + indices.
// ===========================================================================
//
// Oracle (run from /tmp), query point (0.2,0.1) has ALL-DISTINCT distances:
//   python3 -c "import numpy as np; from sklearn.neighbors import KDTree; \
//   d,i=KDTree(np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.],[10.,10.]])) \
//   .query(np.array([[0.2,0.1]]),k=2); print(d.tolist(),i.tolist())"
//   -> [[0.223606797749979, 0.806225774829855]] [[0, 1]]
//   ...k=3 -> [[0.223606797749979, 0.806225774829855, 0.9219544457292888]]
//             [[0, 1, 2]]
#[test]
fn green_query_k2_distinct_distances_and_indices() {
    let data = five_points();
    let tree = KdTree::build(&data);

    // k=2: distinct order [0, 1].
    let n2 = tree.query(&data, &[0.2, 0.1], 2);
    assert_eq!(n2.len(), 2, "k=2 should return 2 neighbors");
    assert_eq!(n2[0].0, 0, "k=2 nearest index");
    assert_eq!(n2[1].0, 1, "k=2 second index");
    assert!(
        (n2[0].1 - 0.223_606_797_749_979).abs() < 1e-12,
        "k=2 nearest distance: got {}",
        n2[0].1
    );
    assert!(
        (n2[1].1 - 0.806_225_774_829_855).abs() < 1e-12,
        "k=2 second distance: got {}",
        n2[1].1
    );

    // k=3: distinct order [0, 1, 2].
    let n3 = tree.query(&data, &[0.2, 0.1], 3);
    assert_eq!(n3.len(), 3, "k=3 should return 3 neighbors");
    let idx: Vec<usize> = n3.iter().map(|n| n.0).collect();
    assert_eq!(idx, vec![0, 1, 2], "k=3 distinct index order");
    assert!(
        (n3[2].1 - 0.921_954_445_729_288_8).abs() < 1e-12,
        "k=3 third distance: got {}",
        n3[2].1
    );
}

// ===========================================================================
// GREEN 2 — REQ-2 shipped: k=1 nearest-neighbor value match.
// ===========================================================================
//
// Oracle (run from /tmp), (0.1,0.1) — index 0 is UNIQUELY nearest (no tie at
// k=1):
//   python3 -c "import numpy as np; from sklearn.neighbors import KDTree; \
//   d,i=KDTree(np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.],[10.,10.]])) \
//   .query(np.array([[0.1,0.1]]),k=1); print(d.tolist(),i.tolist())"
//   -> [[0.14142135623730953]] [[0]]
#[test]
fn green_query_k1_nearest_neighbor() {
    let data = five_points();
    let tree = KdTree::build(&data);
    let neighbors = tree.query(&data, &[0.1, 0.1], 1);

    assert_eq!(neighbors.len(), 1, "k=1 should return 1 neighbor");
    assert_eq!(neighbors[0].0, 0, "nearest index is 0");
    assert!(
        (neighbors[0].1 - 0.141_421_356_237_309_53).abs() < 1e-12,
        "nearest distance: got {}",
        neighbors[0].1
    );
}

// ===========================================================================
// GREEN 3 — REQ-2/REQ-4: k=4 tie case. Assert the SET of indices + the sorted
// distances, NOT a fixed permutation (sklearn's tie order is leaf_size /
// traversal dependent — `:1191`; the design doc records [0,1,2,3] at
// leaf_size=40 vs [0,2,1,3] at leaf_size 1/2).
// ===========================================================================
//
// Oracle (run from /tmp, default leaf_size=40):
//   python3 -c "import numpy as np; from sklearn.neighbors import KDTree; \
//   d,i=KDTree(np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.],[10.,10.]])) \
//   .query(np.array([[0.5,0.5]]),k=4); print(d.tolist(),i.tolist())"
//   -> [[0.7071067811865476, 0.7071067811865476, 0.7071067811865476,
//       0.7071067811865476]] [[0, 1, 2, 3]]
#[test]
fn green_query_k4_tie_set_and_sorted_distances() {
    let data = five_points();
    let tree = KdTree::build(&data);
    let neighbors = tree.query(&data, &[0.5, 0.5], 4);

    assert_eq!(neighbors.len(), 4, "k=4 should return 4 neighbors");

    // Neighbor SET is invariant: {0,1,2,3}; index 4 (at (10,10)) excluded.
    let mut idx: Vec<usize> = neighbors.iter().map(|n| n.0).collect();
    idx.sort_unstable();
    assert_eq!(
        idx,
        vec![0, 1, 2, 3],
        "tie neighbor set must be {{0,1,2,3}}"
    );

    // Every distance is sqrt(0.5) (multiset of distances is invariant).
    for (i, n) in neighbors.iter().enumerate() {
        assert!(
            (n.1 - SQRT_HALF).abs() < 1e-12,
            "neighbor {i} distance: got {} expected {}",
            n.1,
            SQRT_HALF
        );
    }
}

// NOTE: query k > n_samples error contract (REQ-7) is NOT-STARTED (#831),
// blocked on threading `KdTree::query -> Result` through its consumers
// (knn.rs / nearest_neighbors.rs, which need their own design docs +
// Result-returning helpers). sklearn raises ValueError
// (_binary_tree.pxi.tp:1140-1142); ferrolearn silently truncates. No failing
// test committed until the consumer-threading prereq lands.
