//! Adversarial divergence pins for `ferrolearn-neighbors/src/balltree.rs`
//! (`pub struct BallTree`, `BallTree::build` / `build_with_leaf_size` /
//! `query` / `within_radius`) against the live scikit-learn 1.5.2 oracle
//! (`from sklearn.neighbors import BallTree`).
//!
//! Every expected value below is captured from a live `python3 -c "..."` run of
//! sklearn 1.5.2 (R-CHAR-3 — never literal-copied from the ferrolearn side).
//! The exact oracle call and its `(d, i)` output is quoted above each assertion.
//!
//! Design doc: `.design/neighbors/balltree.md` (commit 9d9c63a2). Upstream:
//! `sklearn/neighbors/_ball_tree.pyx.tp` (centroid+radius node bounds,
//! `min_dist`/`min_rdist`) + `sklearn/neighbors/_binary_tree.pxi.tp` (shared
//! `BinaryTree.__init__` / `query` / `query_radius`).
//!
//! ferrolearn API under test:
//!   * `BallTree::build<F: Float + ..>(data: &Array2<F>) -> BallTree`
//!   * `BallTree::build_with_leaf_size<F>(data: &Array2<F>, leaf_size: usize)`
//!   * `BallTree::query<F>(&self, _data: &Array2<F>, query: &[f64], k: usize)
//!     -> Vec<(usize, f64)>` — flat `(index, true-euclidean-dist)` pairs sorted
//!     ascending. Returns a `Vec`, NOT a `Result`; on `k > n_samples` it
//!     silently returns `min(k, n)` neighbors rather than raising.
//!   * `BallTree::within_radius<F>(&self, query: &[f64], radius: f64)
//!     -> Vec<(usize, f64)>` — `(index, true-distance)` pairs, UNSORTED.
//!
//! GREEN guards (must PASS — REQ-1/REQ-2/REQ-3 SHIPPED value contracts):
//!   * `green_query_k1_k2_k3_distinct_distances_and_indices` — REQ-1 / #854.
//!   * `green_query_k1_nearest_neighbor` — REQ-1.
//!   * `green_query_k4_tie_set_and_sorted_distances` — REQ-2.
//!   * `green_within_radius_set_match` — REQ-3.
//!   * `green_leaf_size_invariance` — REQ-1/REQ-2 leaf_size-invariance.
//!
//! ==========================================================================
//! NOT-STARTED divergences — DOCUMENTED ONLY, no forced failing test committed.
//! ==========================================================================
//! Following the kdtree sibling precedent (`divergence_kdtree.rs`), the
//! divergences below are NOT pinned with a committed RED test because they are
//! blocked on prereqs that the discriminator may not author (a fix requires the
//! generator), or are inexpressible through the current public API:
//!
//!   * **k > n_samples** (#858): sklearn `BallTree([[0,0],[1,1]]).query(
//!     [[0,0]], k=5)` raises `ValueError("k must be less than or equal to the
//!     number of training points")` (`_binary_tree.pxi.tp:1140-1142`).
//!     ferrolearn `query` returns a `Vec` (not `Result`) and silently returns
//!     `min(k, n)` neighbors. NOT-STARTED — blocked on threading `query ->
//!     Result` through the non-test consumers `knn.rs` / `nearest_neighbors.rs`
//!     / `radius_neighbors.rs` (same class as kdtree #831). No failing test is
//!     committed until that consumer-threading prereq lands.
//!
//!   * **empty X** (#858): sklearn `BallTree(np.empty((0,2)))` raises
//!     `ValueError("Found array with 0 sample(s) ...")`
//!     (`_binary_tree.pxi.tp:855-856` via `check_array`). ferrolearn `build`
//!     returns an empty tree. Not expressible: `build` returns `BallTree`, not
//!     `Result`, and the empty state is a private field. No test.
//!
//!   * **metric='manhattan'** (#857), **metric default / `p`** (#855),
//!     **query_radius full surface** — `count_only` / `return_distance` /
//!     `sort_results` toggles + mutual-exclusion `ValueError`s (#856),
//!     **batched `(n_query, n_features)` query** (#859),
//!     **`kernel_density`** (#860), **`two_point_correlation`** (#861),
//!     **PyO3 `BallTree` shim** (#862), **ferray substrate** (#863): all
//!     missing features / surface. A missing method is not runtime-pinnable
//!     without inventing surface the impl does not have. No forced tests.

use ferrolearn_neighbors::balltree::BallTree;
use ndarray::{Array2, array};

/// The fixed dataset used across the oracle calls below.
/// `X = [[0,0],[1,0],[0,1],[1,1],[10,10]]` (n_samples = 5).
fn five_points() -> Array2<f64> {
    array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [10.0, 10.0]]
}

// sqrt(0.5) — the common tie distance from (0.5,0.5) to all four unit-square
// corners. Live oracle reports 0.7071067811865476 (see GREEN 3).
const SQRT_HALF: f64 = std::f64::consts::FRAC_1_SQRT_2;

// ===========================================================================
// GREEN 1 — REQ-1 (SHIPPED): k=1 / k=2 / k=3 distinct-distance value parity.
// ===========================================================================
//
// Query point (0.2,0.1) has ALL-DISTINCT distances, giving a deterministic
// (non-tie) index order. Oracle (run from /tmp):
//   python3 -c "import numpy as np; from sklearn.neighbors import BallTree; \
//   d,i=BallTree(np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.],[10.,10.]])) \
//   .query(np.array([[0.2,0.1]]),k=1); print(d.tolist(),i.tolist())"
//   -> [[0.223606797749979]] [[0]]
//
//   ...k=2 -> [[0.223606797749979, 0.806225774829855]] [[0, 1]]
//
//   ...k=3 -> [[0.223606797749979, 0.806225774829855, 0.9219544457292888]]
//             [[0, 1, 2]]
#[test]
fn green_query_k1_k2_k3_distinct_distances_and_indices() {
    let data = five_points();
    let tree = BallTree::build(&data);

    // Confirm the query is genuinely tie-free at the SHIPPED scale: the three
    // distances are strictly increasing and distinct (not a coincidental
    // equal-distance set).
    let n3 = tree.query(&data, &[0.2, 0.1], 3);
    assert_eq!(n3.len(), 3, "k=3 should return 3 neighbors");
    assert!(
        n3[0].1 < n3[1].1 && n3[1].1 < n3[2].1,
        "fixture must be tie-free: distances {} {} {} not strictly increasing",
        n3[0].1,
        n3[1].1,
        n3[2].1
    );

    // k=1: nearest is index 0 at 0.223606797749979.
    let n1 = tree.query(&data, &[0.2, 0.1], 1);
    assert_eq!(n1.len(), 1, "k=1 should return 1 neighbor");
    assert_eq!(n1[0].0, 0, "k=1 nearest index");
    assert!(
        (n1[0].1 - 0.223_606_797_749_979).abs() < 1e-12,
        "k=1 nearest distance: got {}",
        n1[0].1
    );

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
    let idx: Vec<usize> = n3.iter().map(|n| n.0).collect();
    assert_eq!(idx, vec![0, 1, 2], "k=3 distinct index order");
    assert!(
        (n3[0].1 - 0.223_606_797_749_979).abs() < 1e-12,
        "k=3 first distance: got {}",
        n3[0].1
    );
    assert!(
        (n3[1].1 - 0.806_225_774_829_855).abs() < 1e-12,
        "k=3 second distance: got {}",
        n3[1].1
    );
    assert!(
        (n3[2].1 - 0.921_954_445_729_288_8).abs() < 1e-12,
        "k=3 third distance: got {}",
        n3[2].1
    );
}

// ===========================================================================
// GREEN 2 — REQ-1 (SHIPPED): k=1 nearest-neighbor value match for (0.1,0.1).
// ===========================================================================
//
// Oracle (run from /tmp), (0.1,0.1) — index 0 is UNIQUELY nearest:
//   python3 -c "import numpy as np; from sklearn.neighbors import BallTree; \
//   d,i=BallTree(np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.],[10.,10.]])) \
//   .query(np.array([[0.1,0.1]]),k=1); print(d.tolist(),i.tolist())"
//   -> [[0.14142135623730953]] [[0]]
#[test]
fn green_query_k1_nearest_neighbor() {
    let data = five_points();
    let tree = BallTree::build(&data);
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
// GREEN 3 — REQ-2 (SHIPPED): k=4 tie case. Assert the SET of indices + the
// distances, NOT a fixed permutation (sklearn's tie order is leaf_size /
// traversal dependent — the design doc records [0,1,2,3] at leaf_size=40 vs
// [0,2,1,3] at leaf_size=1/2).
// ===========================================================================
//
// Oracle (run from /tmp, default leaf_size=40):
//   python3 -c "import numpy as np; from sklearn.neighbors import BallTree; \
//   d,i=BallTree(np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.],[10.,10.]])) \
//   .query(np.array([[0.5,0.5]]),k=4); print(d.tolist(),i.tolist())"
//   -> [[0.7071067811865476, 0.7071067811865476, 0.7071067811865476,
//       0.7071067811865476]] [[0, 1, 2, 3]]
#[test]
fn green_query_k4_tie_set_and_sorted_distances() {
    let data = five_points();
    let tree = BallTree::build(&data);
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

// ===========================================================================
// GREEN 4 — REQ-3 (SHIPPED): within_radius SET parity vs query_radius.
// ===========================================================================
//
// Oracle (run from /tmp), default sort_results=False so compare as SETS:
//   python3 -c "import numpy as np; from sklearn.neighbors import BallTree; \
//   i,d=BallTree(np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.],[10.,10.]])) \
//   .query_radius(np.array([[0.2,0.1]]),r=1.5,return_distance=True); \
//   print([a.tolist() for a in i],[a.tolist() for a in d])"
//   -> [[0, 1, 2, 3]] [[0.223606797749979, 0.806225774829855,
//       0.9219544457292888, 1.2041594578792296]]
#[test]
fn green_within_radius_set_match() {
    let data = five_points();
    let tree = BallTree::build(&data);
    let mut results = tree.within_radius(&[0.2, 0.1], 1.5);

    // Compare as a SET (both unsorted): sort a copy by index, and the distance
    // multiset separately. Expected from the live oracle above.
    results.sort_by_key(|r| r.0);

    let idx: Vec<usize> = results.iter().map(|r| r.0).collect();
    assert_eq!(idx, vec![0, 1, 2, 3], "in-radius index set");

    // Distances paired with the index-sorted order map to the oracle's
    // index-sorted distances (index i -> its own distance, both identities).
    let expected = [
        0.223_606_797_749_979,
        0.806_225_774_829_855,
        0.921_954_445_729_288_8,
        1.204_159_457_879_229_6,
    ];
    for (r, e) in results.iter().zip(expected.iter()) {
        assert!(
            (r.1 - e).abs() < 1e-12,
            "in-radius distance for index {}: got {} expected {}",
            r.0,
            r.1,
            e
        );
    }
}

// ===========================================================================
// GREEN 5 — REQ-1/REQ-2 leaf_size-invariance: leaf_size=1 gives the same
// neighbor SET + distances as the default (leaf 40) on the tie-free fixture.
// Proves leaf_size changes tree shape, not query value (design doc :88-90).
// ===========================================================================
//
// Reference values from the same oracle as GREEN 1 (k=3, (0.2,0.1)):
//   -> dist [0.223606797749979, 0.806225774829855, 0.9219544457292888],
//      idx  [0, 1, 2]
#[test]
fn green_leaf_size_invariance() {
    let data = five_points();
    let tree_leaf1 = BallTree::build_with_leaf_size(&data, 1);

    let n3 = tree_leaf1.query(&data, &[0.2, 0.1], 3);
    assert_eq!(n3.len(), 3, "leaf_size=1 k=3 should return 3 neighbors");

    // Tie-free fixture: index order and distances must equal the leaf-40 oracle.
    let idx: Vec<usize> = n3.iter().map(|n| n.0).collect();
    assert_eq!(
        idx,
        vec![0, 1, 2],
        "leaf_size=1 index order matches leaf 40"
    );

    let expected = [
        0.223_606_797_749_979,
        0.806_225_774_829_855,
        0.921_954_445_729_288_8,
    ];
    for (n, e) in n3.iter().zip(expected.iter()) {
        assert!(
            (n.1 - e).abs() < 1e-12,
            "leaf_size=1 distance for index {}: got {} expected {}",
            n.0,
            n.1,
            e
        );
    }
}
