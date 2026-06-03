//! RED divergence pins for `DBSCAN` `labels_` / `core_sample_indices_`
//! (`ferrolearn-cluster/src/dbscan.rs`) against the LIVE scikit-learn 1.5.2
//! oracle (`from sklearn.cluster import DBSCAN`, mirroring
//! `sklearn/cluster/_dbscan.py` + `sklearn/cluster/_dbscan_inner.pyx`).
//!
//! These pin the OVERCLAIM in the `## REQ status` table of `dbscan.rs`:
//! REQ-1 (`labels_` VALUE parity) and REQ-2 (`core_sample_indices_` VALUE
//! parity) are marked SHIPPED with the claim that ferrolearn "VALUE-matches
//! sklearn EXACTLY element-wise" on the Euclidean / no-`sample_weight` path.
//! That claim is FALSE on two fixtures the green guards never exercised.
//!
//! Every expected value is a LIVE `sklearn` 1.5.2 oracle value (computed via
//! `python3 -c "..."` from `/tmp`, quoted above each block) — NEVER copied
//! from the ferrolearn side (goal.md R-CHAR-3).
//!
//! Tracking: #952.
//!
//! ## Root cause (two independent divergences)
//!
//! 1. **Traversal order — `dbscan_inner` is DFS-with-label-at-POP, ferrolearn
//!    is BFS-with-label-at-PUSH.** sklearn `dbscan_inner`
//!    (`sklearn/cluster/_dbscan_inner.pyx:26-39`) pushes a neighbor `v` onto a
//!    LIFO stack only checking `if labels[v] == -1`, and assigns the label when
//!    the point is POPPED (`if labels[i] == -1: labels[i] = label_num`).
//!    ferrolearn `Fit::fit` (`dbscan.rs`) uses a `VecDeque` (FIFO) and assigns
//!    the label at PUSH time (`labels[neighbor] = current_cluster`). sklearn's
//!    labeling is therefore genuinely traversal-order-dependent: even with an
//!    IDENTICAL core set and distance graph, sklearn can split points into
//!    distinct clusters that ferrolearn merges into one (`_dbscan.py:436`,
//!    `dbscan_inner(core_samples, neighborhoods, labels)`).
//!
//! 2. **Neighbor-radius boundary — `radius_neighbors` vs `<= eps^2`.** sklearn
//!    routes neighbor search through
//!    `NearestNeighbors(radius=self.eps).radius_neighbors(X)`
//!    (`_dbscan.py:411-422`); ferrolearn `region_query` (`dbscan.rs`) compares
//!    `squared_euclidean <= eps*eps`. For a pair at distance numerically
//!    `0.9999999999999998`, sklearn's tree-computed distance EXCLUDES the
//!    neighbor while ferrolearn's squared-distance INCLUDES it, flipping a
//!    point's core status and the resulting `core_sample_indices_` + `labels_`.

use ferrolearn_cluster::DBSCAN;
use ferrolearn_core::traits::Fit;
use ndarray::Array2;

// ===========================================================================
// RED — Fixture T (PURE traversal divergence: core set IDENTICAL).
//
// 8 points, eps=1.3, min_samples=2. ALL 8 points are core on BOTH sides
// (core_sample_indices_ = [0,1,2,3,4,5,6,7] identically), so this isolates the
// `dbscan_inner` DFS-at-pop vs ferrolearn BFS-at-push label assignment. sklearn
// splits into TWO clusters; ferrolearn merges everything into ONE.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "import numpy as np; from sklearn.cluster import DBSCAN
//     X=np.array([[2.68,0.7],[0.79,2.49],[0.63,1.13],[2.11,1.36],
//                 [0.67,0.87],[2.94,0.2],[2.98,0.51],[0.91,1.86]])
//     m=DBSCAN(eps=1.3,min_samples=2).fit(X)
//     print(m.labels_.tolist(), m.core_sample_indices_.tolist())"
//   ->  [0, 1, 1, 0, 1, 0, 0, 1]   [0, 1, 2, 3, 4, 5, 6, 7]
//
// ferrolearn (observed) -> labels [0,0,0,0,0,0,0,0], core [0..8] (one cluster).
// ===========================================================================

/// Divergence: ferrolearn `DBSCAN::fit().labels()` diverges from
/// `sklearn/cluster/_dbscan.py:436` (`dbscan_inner`, DFS-at-pop) for the
/// 8-point eps=1.3 fixture. sklearn `labels_` = `[0,1,1,0,1,0,0,1]` (two
/// clusters); ferrolearn returns `[0,0,0,0,0,0,0,0]` (one cluster) because its
/// BFS-at-push expansion is not the same traversal sklearn uses. Core set is
/// IDENTICAL on both sides, so this is pure label/tie-break divergence.
/// Tracking: #952.
#[test]
fn red_dbscan_labels_pure_traversal_split() {
    // LIVE sklearn 1.5.2 oracle (quoted above).
    let sklearn_labels: [isize; 8] = [0, 1, 1, 0, 1, 0, 0, 1];

    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            2.68, 0.7, 0.79, 2.49, 0.63, 1.13, 2.11, 1.36, 0.67, 0.87, 2.94, 0.2, 2.98, 0.51, 0.91,
            1.86,
        ],
    )
    .unwrap();

    let fitted = DBSCAN::<f64>::new(1.3)
        .with_min_samples(2)
        .fit(&x, &())
        .unwrap();
    let labels = fitted.labels();

    for (i, &exp) in sklearn_labels.iter().enumerate() {
        assert_eq!(
            labels[i], exp,
            "Fixture T label[{i}]: ferrolearn {} vs sklearn {exp} \
             (dbscan_inner DFS-at-pop splits clusters; ferrolearn BFS-at-push \
             merges them — identical core set, pure traversal divergence)",
            labels[i]
        );
    }
}

// ===========================================================================
// RED — Fixture U (boundary + traversal: core set ALSO diverges).
//
// 9 points, eps=1.0, min_samples=3. idx 7 = [2.72,2.01] and idx 5 = [2.44,1.05]
// lie at numeric distance 0.9999999999999998. sklearn's
// `radius_neighbors(radius=1.0)` EXCLUDES idx5 from idx7's neighborhood
// (idx7 gets [4,7], n=2 < 3 -> NOT core), while ferrolearn `squared_euclidean
// <= eps*eps` INCLUDES it (idx7 gets [4,5,7], n=3 -> CORE). This flips
// core_sample_indices_ AND merges sklearn's two clusters into one.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "import numpy as np; from sklearn.cluster import DBSCAN
//     X=np.array([[2.21,0.14],[1.78,2.79],[2.34,1.0],[0.41,0.28],[2.44,2.73],
//                 [2.44,1.05],[1.51,1.17],[2.72,2.01],[2.31,0.61]])
//     m=DBSCAN(eps=1.0,min_samples=3).fit(X)
//     print(m.labels_.tolist(), m.core_sample_indices_.tolist())"
//   ->  [0, 1, 0, -1, 1, 0, 0, 1, 0]   [0, 2, 4, 5, 6, 8]
//
// ferrolearn (observed) -> labels [0,0,0,-1,0,0,0,0,0], core [0,2,4,5,6,7,8].
// ===========================================================================

/// Divergence: ferrolearn `core_sample_indices()` diverges from
/// `sklearn/cluster/_dbscan.py:438` (`np.where(core_samples)[0]`, where
/// `core_samples` comes from `radius_neighbors`, `:411-422`) for the 9-point
/// eps=1.0 fixture. sklearn core = `[0,2,4,5,6,8]` (idx7 NOT core, since its
/// `radius_neighbors` distance to idx5 = 0.9999999999999998 is excluded);
/// ferrolearn `region_query` `squared_euclidean <= eps*eps` marks idx7 core,
/// giving `[0,2,4,5,6,7,8]`. Tracking: #952.
#[test]
fn red_dbscan_core_indices_radius_boundary() {
    // LIVE sklearn 1.5.2 oracle (quoted above).
    let sklearn_core: [usize; 6] = [0, 2, 4, 5, 6, 8];

    let x = Array2::from_shape_vec(
        (9, 2),
        vec![
            2.21, 0.14, 1.78, 2.79, 2.34, 1.0, 0.41, 0.28, 2.44, 2.73, 2.44, 1.05, 1.51, 1.17,
            2.72, 2.01, 2.31, 0.61,
        ],
    )
    .unwrap();

    let fitted = DBSCAN::<f64>::new(1.0)
        .with_min_samples(3)
        .fit(&x, &())
        .unwrap();

    assert_eq!(
        fitted.core_sample_indices(),
        &sklearn_core,
        "Fixture U core_sample_indices_: ferrolearn {:?} vs sklearn {:?} \
         (idx7 at boundary distance 0.9999999999999998 is excluded by \
         sklearn radius_neighbors but included by ferrolearn squared <= eps^2)",
        fitted.core_sample_indices(),
        sklearn_core
    );
}

/// Divergence: ferrolearn `DBSCAN::fit().labels()` diverges from
/// `sklearn/cluster/_dbscan.py:439` for the same 9-point eps=1.0 fixture.
/// sklearn `labels_` = `[0,1,0,-1,1,0,0,1,0]` (two clusters); ferrolearn
/// returns `[0,0,0,-1,0,0,0,0,0]` (one cluster) — the spurious idx7 core (from
/// the radius-boundary divergence) bridges what sklearn keeps as two separate
/// clusters. Tracking: #952.
#[test]
fn red_dbscan_labels_radius_boundary_merge() {
    // LIVE sklearn 1.5.2 oracle (quoted above).
    let sklearn_labels: [isize; 9] = [0, 1, 0, -1, 1, 0, 0, 1, 0];

    let x = Array2::from_shape_vec(
        (9, 2),
        vec![
            2.21, 0.14, 1.78, 2.79, 2.34, 1.0, 0.41, 0.28, 2.44, 2.73, 2.44, 1.05, 1.51, 1.17,
            2.72, 2.01, 2.31, 0.61,
        ],
    )
    .unwrap();

    let fitted = DBSCAN::<f64>::new(1.0)
        .with_min_samples(3)
        .fit(&x, &())
        .unwrap();
    let labels = fitted.labels();

    for (i, &exp) in sklearn_labels.iter().enumerate() {
        assert_eq!(
            labels[i], exp,
            "Fixture U label[{i}]: ferrolearn {} vs sklearn {exp} \
             (ferrolearn merges sklearn's two clusters into one)",
            labels[i]
        );
    }
}
