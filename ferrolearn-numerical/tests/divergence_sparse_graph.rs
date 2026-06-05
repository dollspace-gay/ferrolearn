//! Divergence tests for `ferrolearn-numerical/src/sparse_graph.rs` vs the LIVE
//! scipy 1.17.1 `scipy.sparse.csgraph` oracle (crosslink translation unit #1947).
//!
//! All expected values are computed by a live scipy call run from `/tmp`
//! (R-CHAR-3 — NEVER copied from the ferrolearn side). The oracle commands and
//! their outputs are quoted in each test's doc-comment so the fixer's target is
//! unambiguous.
//!
//! Contents:
//!   * ONE FAILING pin: `mst_representation_upper_triangular` (#1948, the
//!     single-file-fixable MST-representation divergence). RED now, goes green
//!     when the fixer drops the mirror push in `minimum_spanning_tree`.
//!   * GREEN guards for the three SHIPPED behaviors (dijkstra single-source,
//!     dijkstra all-pairs, connected_components count+labels) — these PASS now
//!     and guard against regression.
//!
//! The remaining NOT-STARTED REQs (#1949 signature/flags, #1950 predecessor
//! sentinel, #1951 cc directed/strong, #1952 missing fns, #1953 error type,
//! #1954 no consumer, #1955 ferray substrate) are structural blockers that a
//! single-file fix cannot close this iteration (R-DEFER-3); they are NOT pinned
//! here as doomed tests.

use ferrolearn_numerical::sparse_graph::{
    connected_components, dijkstra, dijkstra_all_pairs, minimum_spanning_tree,
};
use sprs::{CsMat, TriMat};

/// Build a CSR matrix from `(row, col, value)` triplets.
fn csr(n: usize, triplets: &[(usize, usize, f64)]) -> CsMat<f64> {
    let mut tri = TriMat::new((n, n));
    for &(r, c, v) in triplets {
        tri.add_triplet(r, c, v);
    }
    tri.to_csr()
}

/// The canonical symmetric 4-node weight matrix used by the design doc / oracle:
/// `G = [[0,1,0,4],[1,0,2,0],[0,2,0,3],[4,0,3,0]]`.
fn graph4() -> CsMat<f64> {
    csr(
        4,
        &[
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 2, 2.0),
            (2, 1, 2.0),
            (2, 3, 3.0),
            (3, 2, 3.0),
            (0, 3, 4.0),
            (3, 0, 4.0),
        ],
    )
}

// ===========================================================================
// (1) FAILING PIN — REQ-MST-REPR (#1948, single-file-fixable, HEADLINE)
// ===========================================================================

/// Divergence: ferrolearn's `minimum_spanning_tree` in
/// `ferrolearn-numerical/src/sparse_graph.rs` diverges from
/// `scipy.sparse.csgraph.minimum_spanning_tree`
/// (`scipy/sparse/csgraph/__init__.py:79` — "an undirected graph is represented
/// by a symmetric matrix"; the MST is returned as a *directed* spanning-tree,
/// one entry per edge).
///
/// ferrolearn stores BOTH directions per kept edge
/// (`mst_triplets.push((u, v, w)); mst_triplets.push((v, u, w));`) → a SYMMETRIC
/// CSR with `nnz = 2*n_edges` and entry-sum `= 2*(MST weight)`. scipy returns
/// ONE entry per edge, UPPER-TRIANGULAR (`i < j`) for a symmetric input.
///
/// The selected EDGE SET and weights are correct (edges (0,1)=1, (1,2)=2,
/// (2,3)=3, total 6); ONLY the both-directions storage diverges.
///
/// LIVE ORACLE (scipy 1.17.1, run from /tmp):
/// ```text
/// $ python3 -c "import numpy as np, scipy.sparse as sp;
///   from scipy.sparse.csgraph import minimum_spanning_tree as m;
///   g=sp.csr_matrix(np.array([[0,1,0,4],[1,0,2,0],[0,2,0,3],[4,0,3,0]],float));
///   print(m(g).toarray().tolist(), m(g).nnz, m(g).sum())"
/// [[0.0,1.0,0.0,0.0],[0.0,0.0,2.0,0.0],[0.0,0.0,0.0,3.0],[0.0,0.0,0.0,0.0]] 3 6.0
/// ```
/// scipy: nnz=3, sum=6.0, upper-triangular. ferrolearn: nnz=6, sum=12.0, symmetric.
///
/// Orientation convention (`i < j`) confirmed on a SECOND symmetric graph
/// (5-node `A = [[0,2,0,6,0],[2,0,3,8,5],[0,3,0,0,7],[6,8,0,0,9],[0,5,7,9,0]]`):
/// ```text
/// $ python3 -c "... print(m(g5).toarray().tolist(), m(g5).nnz, m(g5).sum())"
/// [[0,2,0,6,0],[0,0,3,0,5],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]] 4 16.0
/// ```
/// Every emitted entry is `i < j` (edges (0,1)=2,(0,3)=6,(1,2)=3,(1,4)=5) — the
/// convention is consistently upper-triangular `i < j`, one entry per edge.
///
/// Tracking: #1948.
#[test]
fn mst_representation_upper_triangular() {
    let mst = minimum_spanning_tree(&graph4()).expect("MST should succeed");

    // (a) one entry per edge — scipy nnz = 3 (ferrolearn yields 6).
    assert_eq!(
        mst.nnz(),
        3,
        "scipy MST nnz=3 (one entry per edge); ferrolearn stores both directions"
    );

    // (b) entry-sum == MST weight = 6.0 (ferrolearn yields 12.0 = 2*weight).
    let sum: f64 = mst.iter().map(|(&w, _)| w).sum();
    assert!(
        (sum - 6.0).abs() < 1e-12,
        "scipy MST entry-sum = 6.0 (MST weight); got {sum}"
    );

    // (c) dense matrix equals scipy's upper-triangular `i < j` orientation.
    //     scipy: [[0,1,0,0],[0,0,2,0],[0,0,0,3],[0,0,0,0]].
    let dense = mst.to_dense();
    let expected = [
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0, 3.0],
        [0.0, 0.0, 0.0, 0.0],
    ];
    for i in 0..4 {
        for j in 0..4 {
            assert!(
                (dense[[i, j]] - expected[i][j]).abs() < 1e-12,
                "MST[{i},{j}] = {} but scipy upper-triangular = {}",
                dense[[i, j]],
                expected[i][j]
            );
        }
    }
}

// ===========================================================================
// (2) GREEN GUARDS — SHIPPED behaviors (must PASS)
// ===========================================================================

/// REQ-DIJ-DIST green guard: single-source distances match scipy element-wise.
///
/// LIVE ORACLE (scipy 1.17.1, run from /tmp):
/// ```text
/// $ python3 -c "... print(dijkstra(g, indices=0).tolist())"  ->  [0.0, 1.0, 3.0, 4.0]
/// ```
#[test]
fn dijkstra_single_source_matches_scipy() {
    let res = dijkstra(&graph4(), 0).expect("dijkstra should succeed");
    let expected = [0.0_f64, 1.0, 3.0, 4.0];
    for (i, &e) in expected.iter().enumerate() {
        assert!(
            (res.distances[i] - e).abs() < 1e-12,
            "dijkstra src0 dist[{i}] = {}; scipy = {e}",
            res.distances[i]
        );
    }
}

/// REQ-DIJ-DIST green guard: unreachable vertices are `inf`, matching scipy.
///
/// LIVE ORACLE — disconnected `{0,1}|{2,3}`,
/// `C = [[0,1,0,0],[1,0,0,0],[0,0,0,2],[0,0,2,0]]`:
/// ```text
/// $ python3 -c "... print(dijkstra(gc, indices=0).tolist())"  ->  [0.0, 1.0, inf, inf]
/// ```
#[test]
fn dijkstra_disconnected_inf_matches_scipy() {
    let g = csr(4, &[(0, 1, 1.0), (1, 0, 1.0), (2, 3, 2.0), (3, 2, 2.0)]);
    let res = dijkstra(&g, 0).expect("dijkstra should succeed");
    assert!((res.distances[0] - 0.0).abs() < 1e-12);
    assert!((res.distances[1] - 1.0).abs() < 1e-12);
    assert!(res.distances[2].is_infinite(), "vertex 2 should be inf");
    assert!(res.distances[3].is_infinite(), "vertex 3 should be inf");
}

/// REQ-ALLPAIRS green guard: full `n x n` matrix matches `scipy.dijkstra(g)`.
///
/// LIVE ORACLE (scipy 1.17.1, run from /tmp):
/// ```text
/// $ python3 -c "... print(dijkstra(g).tolist())"
/// [[0,1,3,4],[1,0,2,5],[3,2,0,3],[4,5,3,0]]
/// ```
#[test]
fn dijkstra_all_pairs_matches_scipy() {
    let dist = dijkstra_all_pairs(&graph4()).expect("all-pairs should succeed");
    let expected = [
        [0.0, 1.0, 3.0, 4.0],
        [1.0, 0.0, 2.0, 5.0],
        [3.0, 2.0, 0.0, 3.0],
        [4.0, 5.0, 3.0, 0.0],
    ];
    for i in 0..4 {
        for j in 0..4 {
            assert!(
                (dist[[i, j]] - expected[i][j]).abs() < 1e-12,
                "allpairs[{i},{j}] = {}; scipy = {}",
                dist[[i, j]],
                expected[i][j]
            );
        }
    }
}

/// REQ-CC-LABELS green guard: component count AND the exact per-vertex label
/// array match `scipy.connected_components(g, directed=False)`.
///
/// LIVE ORACLE — `{0,1,2}|{3,4}`:
/// ```text
/// $ python3 -c "... print(connected_components(gb, directed=False))"
/// (2, array([0, 0, 0, 1, 1], dtype=int32))
/// ```
#[test]
fn connected_components_labels_match_scipy() {
    // {0,1,2} fully connected, {3,4} connected.
    let g = csr(
        5,
        &[
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 2, 1.0),
            (2, 1, 1.0),
            (0, 2, 1.0),
            (2, 0, 1.0),
            (3, 4, 1.0),
            (4, 3, 1.0),
        ],
    );
    let res = connected_components(&g).expect("cc should succeed");
    assert_eq!(res.n_components, 2, "scipy n_components = 2");
    // scipy label array: [0,0,0,1,1] (ascending node-discovery order).
    let expected = [0usize, 0, 0, 1, 1];
    for (i, &e) in expected.iter().enumerate() {
        assert_eq!(
            res.labels[i], e,
            "cc label[{i}] = {}; scipy = {e}",
            res.labels[i]
        );
    }
}

/// REQ-CC-LABELS green guard, second graph — `{0,1}|{2,3}`.
///
/// LIVE ORACLE:
/// ```text
/// $ python3 -c "... print(connected_components(gc, directed=False))"
/// (2, array([0, 0, 1, 1], dtype=int32))
/// ```
#[test]
fn connected_components_labels_match_scipy_pair() {
    let g = csr(4, &[(0, 1, 1.0), (1, 0, 1.0), (2, 3, 1.0), (3, 2, 1.0)]);
    let res = connected_components(&g).expect("cc should succeed");
    assert_eq!(res.n_components, 2, "scipy n_components = 2");
    let expected = [0usize, 0, 1, 1];
    for (i, &e) in expected.iter().enumerate() {
        assert_eq!(
            res.labels[i], e,
            "cc label[{i}] = {}; scipy = {e}",
            res.labels[i]
        );
    }
}
