//! Adversarial divergence pins for `ferrolearn-neighbors/src/graph.rs` against
//! the live scikit-learn 1.5.2 oracle.
//!
//! Every expected value below is captured from a live `python3 -c "..."` run of
//! sklearn 1.5.2 (R-CHAR-3 — never literal-copied from the ferrolearn side).
//! The exact oracle call and its `.toarray()` / `.data`/`.indices` output is
//! quoted above each assertion.
//!
//! Design doc: `.design/neighbors/graph.md` (REQ-1/REQ-2/REQ-3/REQ-5).
//! Upstream: `sklearn/neighbors/_graph.py` (`_query_include_self`:34,
//! `kneighbors_graph`:59, `radius_neighbors_graph`:164) and
//! `sklearn/neighbors/_base.py` (`sort_graph_by_row_values`:201, :271-289).
//!
//! RED pins (must FAIL against current graph.rs — deterministic divergences):
//!   * `divergence_kneighbors_graph_excludes_self_*`  — blocker #826/#823
//!   * `divergence_radius_neighbors_graph_excludes_self_edge` — blocker #824
//!   * `divergence_sort_graph_by_row_values_sorts_by_value` — blocker #825
//!
//! GREEN guards (must PASS — shipped value/structure contract, REQ-4/REQ-7):
//!   * `green_connectivity_offdiagonal_edge_is_one`
//!   * `green_distance_offdiagonal_edge_is_sqrt2`
//!   * `green_kneighbors_graph_csr_shape_is_n_by_n`

use ferrolearn_neighbors::{GraphMode, kneighbors_graph, radius_neighbors_graph};
use ferrolearn_sparse::CsrMatrix;
use ndarray::{Array2, array};

/// `X = [[0,0],[1,1],[2,2]]` — the first example in sklearn's own docstring.
fn diagonal_three() -> Array2<f64> {
    array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
}

// √2 — the sklearn euclidean distance between adjacent points [[0,0],[1,1]].
// python3 -c "import math;print(repr(math.sqrt(2.0)))" -> 1.4142135623730951
const SQRT2: f64 = std::f64::consts::SQRT_2;

// ===========================================================================
// RED 1 — kneighbors_graph default include_self=False (HEADLINE, REQ-1/5).
// ===========================================================================

/// Divergence: `kneighbors_graph` in `ferrolearn-neighbors/src/graph.rs`
/// (queries `nn.kneighbors(x, …)` against `x` itself with NO self-exclusion)
/// diverges from `sklearn/neighbors/_graph.py:59` whose default
/// `include_self=False` routes through `_query_include_self` (`:34`:
/// "it does not include each sample as its own neighbors / if not include_self:
/// X = None") so each point's OWN row is excluded — the graph has a zero
/// diagonal and reports `n_neighbors` NON-self neighbors per row.
///
/// Oracle (live sklearn 1.5.2, run from /tmp):
///   python3 -c "from sklearn.neighbors import kneighbors_graph; \
///   print(kneighbors_graph([[0,0],[1,1],[2,2]],1,mode='connectivity').toarray().tolist())"
///   -> [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
///
/// ferrolearn includes self (the 0-distance diagonal) → expected to differ.
/// Tracking: #826 (underlies #823).
#[test]
fn divergence_kneighbors_graph_excludes_self_connectivity() {
    let x = diagonal_three();
    let g: CsrMatrix<f64> = kneighbors_graph(&x, 1, GraphMode::Connectivity).unwrap();
    let dense = g.to_dense();

    // sklearn .toarray() — zero diagonal.
    let expected = array![[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    assert_eq!(
        dense, expected,
        "kneighbors_graph(X,1,connectivity) must equal sklearn's zero-diagonal \
         graph [[0,1,0],[1,0,0],[0,1,0]] (_graph.py:59 default include_self=False)"
    );
}

/// Divergence: distance mode of the same default-exclude-self contract.
///
/// Oracle (live sklearn 1.5.2, run from /tmp):
///   python3 -c "from sklearn.neighbors import kneighbors_graph; \
///   print(kneighbors_graph([[0,0],[1,1],[2,2]],1,mode='distance').toarray().tolist())"
///   -> [[0.0, 1.4142135623730951, 0.0],
///       [1.4142135623730951, 0.0, 0.0],
///       [0.0, 1.4142135623730951, 0.0]]
///
/// ferrolearn's nearest neighbor of each row is the row itself at distance 0.0,
/// so the off-diagonal √2 edges and the zero diagonal both differ.
/// Tracking: #823.
#[test]
fn divergence_kneighbors_graph_excludes_self_distance() {
    let x = diagonal_three();
    let g: CsrMatrix<f64> = kneighbors_graph(&x, 1, GraphMode::Distance).unwrap();
    let dense = g.to_dense();

    let expected = array![[0.0, SQRT2, 0.0], [SQRT2, 0.0, 0.0], [0.0, SQRT2, 0.0]];
    assert_eq!(
        dense, expected,
        "kneighbors_graph(X,1,distance) must equal sklearn's zero-diagonal \
         √2-edge graph (_graph.py:59 default include_self=False)"
    );
}

// ===========================================================================
// RED 2 — radius_neighbors_graph default include_self=False (REQ-2).
// ===========================================================================

/// Divergence: `radius_neighbors_graph` in `ferrolearn-neighbors/src/graph.rs`
/// (queries `x` against itself; `radius_to_csr` dedups columns but never drops
/// the self-edge) diverges from `sklearn/neighbors/_graph.py:164` whose default
/// `include_self=False` (`:255` `_query_include_self(...)` → `X=None`) excludes
/// the always-within-radius distance-0 self-edge — zero diagonal.
///
/// Oracle (live sklearn 1.5.2, run from /tmp):
///   python3 -c "from sklearn.neighbors import radius_neighbors_graph; \
///   print(radius_neighbors_graph([[0,0],[1,1],[2,2]],1.5,mode='connectivity').toarray().tolist())"
///   -> [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
///
/// ferrolearn keeps the self-edge → diagonal divergence.
/// Tracking: #824.
#[test]
fn divergence_radius_neighbors_graph_excludes_self_edge() {
    let x = diagonal_three();
    let g: CsrMatrix<f64> = radius_neighbors_graph(&x, 1.5, GraphMode::Connectivity).unwrap();
    let dense = g.to_dense();

    let expected = array![[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]];
    assert_eq!(
        dense, expected,
        "radius_neighbors_graph(X,1.5,connectivity) must equal sklearn's \
         zero-diagonal graph [[0,1,0],[1,0,1],[0,1,0]] (_graph.py:164 default \
         include_self=False)"
    );
}

// NOTE: sort_graph_by_row_values value-sort parity (REQ-3) is NOT-STARTED
// (#825), blocked on ferrolearn-sparse CsrMatrix::new_unchecked (#826) — the
// sprs-backed CsrMatrix rejects the column-unsorted (value-sorted) CSR sklearn
// produces. No failing test committed until the prereq lands.

// ===========================================================================
// GREEN guards — value-mapping + CSR-shape contract that ships TODAY (REQ-4/7).
//
// These probe the [0,1] OFF-diagonal edge with n_neighbors=2: ferrolearn emits
// {self=0, nearest-other=1} for point 0, so the [0,1] cell IS populated and its
// stored VALUE (the connectivity/distance mapping under test) matches sklearn
// even though the diagonal/self-row behavior is wrong in the RED pins above.
// (With n_neighbors=1 ferrolearn returns only the self-edge, so the [0,1] cell
// is empty — which is why these guards must use k=2, not k=1.)
// ===========================================================================

/// Green (REQ-4/AC-7): `GraphMode::Connectivity` stores `1.0` on every emitted
/// edge. The [0,1] off-diagonal entry must be `1.0` — the sklearn `mode`
/// value mapping (`_graph.py:82-85` connectivity = ones), independent of the
/// self-row divergence.
///
/// Oracle (live sklearn 1.5.2, run from /tmp):
///   python3 -c "from sklearn.neighbors import kneighbors_graph; \
///   print(kneighbors_graph([[0,0],[1,1],[2,2]],2,mode='connectivity').toarray().tolist())"
///   -> [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]   ([0,1] == 1.0)
#[test]
fn green_connectivity_offdiagonal_edge_is_one() {
    let x = diagonal_three();
    let g: CsrMatrix<f64> = kneighbors_graph(&x, 2, GraphMode::Connectivity).unwrap();
    let dense = g.to_dense();
    assert_eq!(
        dense[[0, 1]],
        1.0,
        "connectivity edge value must be 1.0 (sklearn mode=connectivity)"
    );
}

/// Green (REQ-4/AC-7): `GraphMode::Distance` stores the euclidean distance on
/// each emitted edge. The [0,1] off-diagonal entry must be √2 — the sklearn
/// `mode='distance'` mapping (minkowski p=2).
///
/// Oracle (live sklearn 1.5.2, run from /tmp):
///   python3 -c "from sklearn.neighbors import kneighbors_graph; \
///   print(kneighbors_graph([[0,0],[1,1],[2,2]],2,mode='distance').toarray().tolist())"
///   -> [[0.0, 1.4142135623730951, 2.8284271247461903], ...]   ([0,1] == √2)
#[test]
fn green_distance_offdiagonal_edge_is_sqrt2() {
    let x = diagonal_three();
    let g: CsrMatrix<f64> = kneighbors_graph(&x, 2, GraphMode::Distance).unwrap();
    let dense = g.to_dense();
    assert_eq!(
        dense[[0, 1]],
        SQRT2,
        "distance edge value must equal euclidean distance √2 (sklearn \
         mode=distance, minkowski p=2)"
    );
}

/// Green (REQ-7): output CSR shape is `(n_samples, n_samples)`. sklearn
/// `kneighbors_graph` returns an `(n_samples, n_samples)` matrix
/// (`_graph.py:59` docstring); for `X` with 3 rows that is 3x3.
#[test]
fn green_kneighbors_graph_csr_shape_is_n_by_n() {
    let x = diagonal_three();
    let g: CsrMatrix<f64> = kneighbors_graph(&x, 1, GraphMode::Connectivity).unwrap();
    assert_eq!(g.n_rows(), 3, "CSR rows == n_samples");
    assert_eq!(g.n_cols(), 3, "CSR cols == n_samples");
}
