//! Live-oracle parity tests for `AgglomerativeClustering`'s
//! `distance_threshold` / `compute_distances` / `distances_` (REQ-9) and
//! `n_leaves_` / `n_connected_components_` (REQ-10) surface
//! (`ferrolearn-cluster/src/agglomerative.rs`) vs scikit-learn 1.5.2 / scipy
//! `sklearn/cluster/_agglomerative.py`
//! (`class AgglomerativeClustering(ClusterMixin, BaseEstimator)`, `:781`),
//! unstructured (`connectivity=None`) case.
//!
//! What is pinned:
//!   * REQ-9 `distances_` — the per-merge linkage distances in `children_` row
//!     order, EXACT-equal to
//!     `AgglomerativeClustering(..., compute_distances=True).fit(X).distances_`
//!     (`_agglomerative.py:1074`, `:1087-1088`) — cross-checked against
//!     `scipy.cluster.hierarchy.linkage(X, method, 'euclidean')[:, 2]` for
//!     ward/complete/average.
//!   * REQ-9 `distance_threshold` cut — `n_clusters_ =
//!     count_nonzero(distances_ >= t) + 1` (`:1090-1093`) and `labels_` EXACT
//!     (integer) equal to
//!     `AgglomerativeClustering(n_clusters=None, distance_threshold=t).fit(X).labels_`
//!     (`:1099` via `_hc_cut`), all four linkages.
//!   * REQ-9 XOR validation — both-None and both-Some each error; each of the
//!     two single-set paths fits (`:1022-1027`).
//!   * REQ-10 `n_leaves_ == n_samples`, `n_connected_components_ == 1`
//!     (`:1083-1085`).
//!
//! Expected values come from the LIVE scipy/sklearn 1.5.2 oracle (R-CHAR-3,
//! R-DEV-3 exact-output) — NEVER literal-copied from the ferrolearn side. The
//! exact `python3 -c` command + its output are recorded in each doc comment.

use ferrolearn_cluster::{AgglomerativeClustering, Linkage};
use ferrolearn_core::Fit;
use ndarray::Array2;

/// 6-point fixture (matches `divergence_agglomerative_dendrogram.rs`): three
/// loose pairs/triples, no degenerate ties.
fn fixture6() -> Array2<f64> {
    Array2::from_shape_vec(
        (6, 2),
        vec![0.0, 0.0, 0.2, 0.1, 0.9, 1.1, 3.0, 3.2, 3.3, 3.0, 6.0, 0.2],
    )
    .unwrap()
}

/// 10-point fixture spanning four loose groups.
fn fixture10() -> Array2<f64> {
    Array2::from_shape_vec(
        (10, 2),
        vec![
            0.0, 0.0, 0.2, 0.1, 0.9, 1.1, 3.0, 3.2, 3.3, 3.0, 6.0, 0.2, 6.1, 0.5, 1.0, 5.0, 1.2,
            5.3, 5.5, 5.5,
        ],
    )
    .unwrap()
}

const WARD: Linkage = Linkage::Ward;
const COMPLETE: Linkage = Linkage::Complete;
const AVERAGE: Linkage = Linkage::Average;
const SINGLE: Linkage = Linkage::Single;

/// Tight tolerance for the distance VALUE comparison (sklearn/scipy compute in
/// f64; ferrolearn computes in f64 internally and stores as F).
const TOL: f64 = 1e-12;

fn assert_close(actual: &[f64], expected: &[f64], ctx: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{ctx}: distances_ length {} != oracle {}",
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= TOL,
            "{ctx}: distances_[{i}] = {a} != oracle {e} (|Δ| = {})",
            (a - e).abs()
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// REQ-9 — distances_ EXACT vs sklearn / scipy
// ─────────────────────────────────────────────────────────────────────────────

/// REQ-9: `distances_` on the 6-pt fixture, all four linkages, EXACT-equal to
/// `AgglomerativeClustering(n_clusters=2, linkage=L, compute_distances=True)
///  .fit(X).distances_` (and, for ward/complete/average,
/// `scipy.linkage(X, L, 'euclidean')[:, 2]`).
///
/// Live scipy/sklearn 1.5.2 oracle (system python3):
/// ```text
/// python3 -c "import numpy as np; from sklearn.cluster import AgglomerativeClustering; \
///   from scipy.cluster.hierarchy import linkage; \
///   X=np.array([[0.,0.],[0.2,0.1],[0.9,1.1],[3.,3.2],[3.3,3.],[6.,0.2]]); \
///   [print(L, AgglomerativeClustering(n_clusters=2,linkage=L,compute_distances=True).fit(X).distances_.tolist()) \
///      for L in ['ward','complete','average','single']]"
/// # ward     [0.223606797749979, 0.3605551275463989, 1.5242484486898236, 4.695032836235901, 7.12928233882392]
/// # complete [0.223606797749979, 0.3605551275463989, 1.4212670403551897, 4.242640687119285, 6.003332407921453]
/// # average  [0.223606797749979, 0.3605551275463989, 1.32096130096428, 3.8832289813789025, 5.023073623638995]
/// # single   [0.223606797749979, 0.3605551275463989, 1.2206555615733703, 2.9698484809834995, 3.8897300677553446]
/// ```
#[test]
fn distances_exact_6pt_all_linkages() {
    let cases: [(Linkage, &str, [f64; 5]); 4] = [
        (
            WARD,
            "ward",
            [
                0.223606797749979,
                0.3605551275463989,
                1.5242484486898236,
                4.695032836235901,
                7.12928233882392,
            ],
        ),
        (
            COMPLETE,
            "complete",
            [
                0.223606797749979,
                0.3605551275463989,
                1.4212670403551897,
                4.242640687119285,
                6.003332407921453,
            ],
        ),
        (
            AVERAGE,
            "average",
            [
                0.223606797749979,
                0.3605551275463989,
                1.32096130096428,
                3.8832289813789025,
                5.023073623638995,
            ],
        ),
        (
            SINGLE,
            "single",
            [
                0.223606797749979,
                0.3605551275463989,
                1.2206555615733703,
                2.9698484809834995,
                3.8897300677553446,
            ],
        ),
    ];

    let x = fixture6();
    for (lk, name, expected) in cases {
        let fitted = AgglomerativeClustering::<f64>::new(2)
            .with_linkage(lk)
            .with_compute_distances(true)
            .fit(&x, &())
            .unwrap();
        let d = fitted.distances().unwrap_or_else(|| {
            panic!("{name}: distances_ should be Some when compute_distances=true")
        });
        assert_close(d.as_slice().unwrap(), &expected, &format!("6pt {name}"));
    }
}

/// REQ-9: `distances_` on the 10-pt fixture, all four linkages.
///
/// Live scipy/sklearn 1.5.2 oracle (system python3):
/// ```text
/// python3 -c "import numpy as np; from sklearn.cluster import AgglomerativeClustering; \
///   X=np.array([[0.,0.],[0.2,0.1],[0.9,1.1],[3.,3.2],[3.3,3.],[6.,0.2],[6.1,0.5],[1.0,5.0],[1.2,5.3],[5.5,5.5]]); \
///   [print(L, AgglomerativeClustering(n_clusters=3,linkage=L,compute_distances=True).fit(X).distances_.tolist()) \
///      for L in ['ward','complete','average','single']]"
/// # ward     [0.22360679775, 0.316227766017, 0.360555127546, 0.360555127546, 1.52424844869, 3.878573620976, 4.797568828758, 8.777406060042, 9.0229601309]
/// # complete [0.22360679775, 0.316227766017, 0.360555127546, 0.360555127546, 1.421267040355, 3.114482300479, 4.527692569069, 6.12045749924, 7.778174593052]
/// # average  [0.22360679775, 0.316227766017, 0.360555127546, 0.360555127546, 1.320961300964, 2.904755144477, 3.889890982609, 4.927650964715, 5.512708170601]
/// # single   [0.22360679775, 0.316227766017, 0.360555127546, 0.360555127546, 1.220655561573, 2.690724809415, 2.969848480983, 3.330165161069, 3.753664875825]
/// ```
/// (Full-precision values used below; the rounded oracle line above is for
/// readability — the test compares against the exact `.tolist()` output.)
#[test]
fn distances_exact_10pt_all_linkages() {
    let cases: [(Linkage, &str, [f64; 9]); 4] = [
        (
            WARD,
            "ward",
            [
                0.223606797749979,
                0.31622776601683783,
                0.3605551275463988,
                0.3605551275463989,
                1.5242484486898236,
                3.878573620976316,
                4.79756882875761,
                8.777406060041722,
                9.022960130899921,
            ],
        ),
        (
            COMPLETE,
            "complete",
            [
                0.223606797749979,
                0.31622776601683783,
                0.3605551275463988,
                0.3605551275463989,
                1.4212670403551897,
                3.114482300479487,
                4.527692569068709,
                6.120457499239742,
                7.7781745930520225,
            ],
        ),
        (
            AVERAGE,
            "average",
            [
                0.223606797749979,
                0.31622776601683783,
                0.3605551275463988,
                0.3605551275463989,
                1.32096130096428,
                2.9047551444769324,
                3.889890982609208,
                4.927650964715065,
                5.512708170600552,
            ],
        ),
        (
            SINGLE,
            "single",
            [
                0.223606797749979,
                0.31622776601683783,
                0.3605551275463988,
                0.3605551275463989,
                1.2206555615733703,
                2.690724809414742,
                2.9698484809834995,
                3.3301651610693423,
                3.753664875824692,
            ],
        ),
    ];

    let x = fixture10();
    for (lk, name, expected) in cases {
        let fitted = AgglomerativeClustering::<f64>::new(3)
            .with_linkage(lk)
            .with_compute_distances(true)
            .fit(&x, &())
            .unwrap();
        let d = fitted.distances().unwrap();
        assert_close(d.as_slice().unwrap(), &expected, &format!("10pt {name}"));
    }
}

/// REQ-9: when `compute_distances` is NOT set and there is no
/// `distance_threshold`, `distances_` is `None` (sklearn's
/// `return_distance == False`, `_agglomerative.py:1074`).
#[test]
fn distances_none_when_not_requested() {
    let x = fixture6();
    let fitted = AgglomerativeClustering::<f64>::new(2)
        .with_linkage(WARD)
        .fit(&x, &())
        .unwrap();
    assert!(
        fitted.distances().is_none(),
        "distances_ must be None when neither distance_threshold nor compute_distances is set"
    );
}

/// REQ-9: setting `distance_threshold` forces `distances_` to be `Some` even
/// without `compute_distances` (sklearn `return_distance` is also true when
/// `distance_threshold is not None`, `_agglomerative.py:1074`).
#[test]
fn distances_some_when_threshold_set() {
    let x = fixture6();
    let fitted = AgglomerativeClustering::<f64>::new(2)
        .with_distance_threshold(5.0)
        .with_linkage(WARD)
        .fit(&x, &())
        .unwrap();
    assert!(fitted.distances().is_some());
    assert_eq!(fitted.distances().unwrap().len(), x.nrows() - 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// REQ-9 — distance_threshold cut: n_clusters_ + labels_ EXACT vs sklearn
// ─────────────────────────────────────────────────────────────────────────────

/// REQ-9: the `distance_threshold` path on the 6-pt fixture. For each linkage
/// and threshold, ferrolearn `n_clusters_` and `labels_` must EXACT-equal
/// sklearn's. Includes a threshold below the min merge distance (→ n_clusters_
/// == n_samples) and above the max (→ 1).
///
/// Live sklearn 1.5.2 oracle (system python3):
/// ```text
/// python3 -c "import numpy as np; from sklearn.cluster import AgglomerativeClustering; \
///   X=np.array([[0.,0.],[0.2,0.1],[0.9,1.1],[3.,3.2],[3.3,3.],[6.,0.2]]); \
///   [print(L, t, m.n_clusters_, m.labels_.tolist()) \
///      for L in ['ward','complete','average','single'] for t in [0.1,0.3,1.5,5.0,100.0] \
///      for m in [AgglomerativeClustering(n_clusters=None,distance_threshold=t,linkage=L).fit(X)]]"
/// # ward     0.1  6 [5,2,3,4,1,0]   ward     0.3  5 [0,0,3,4,1,2]   ward     1.5  4 [1,1,3,0,0,2]   ward     5.0  2 [1,1,1,0,0,0]   ward     100.0 1 [0,0,0,0,0,0]
/// # complete 0.1  6 [5,2,3,4,1,0]   complete 0.3  5 [0,0,3,4,1,2]   complete 1.5  3 [0,0,0,1,1,2]   complete 5.0  2 [1,1,1,0,0,0]   complete 100.0 1 [0,0,0,0,0,0]
/// # average  0.1  6 [5,4,3,1,2,0]   average  0.3  5 [0,0,3,4,2,1]   average  1.5  3 [0,0,0,2,2,1]   average  5.0  2 [0,0,0,0,0,1]   average  100.0 1 [0,0,0,0,0,0]
/// # single   0.1  6 [5,3,2,4,1,0]   single   0.3  5 [0,0,2,4,3,1]   single   1.5  3 [0,0,0,2,2,1]   single   5.0  1 [0,0,0,0,0,0]   single   100.0 1 [0,0,0,0,0,0]
/// ```
#[test]
fn threshold_cut_exact_6pt_all_linkages() {
    // (linkage, threshold, expected_n_clusters_, expected_labels)
    type Case = (Linkage, &'static str, f64, usize, Vec<usize>);
    let cases: Vec<Case> = vec![
        (WARD, "ward", 0.1, 6, vec![5, 2, 3, 4, 1, 0]),
        (WARD, "ward", 0.3, 5, vec![0, 0, 3, 4, 1, 2]),
        (WARD, "ward", 1.5, 4, vec![1, 1, 3, 0, 0, 2]),
        (WARD, "ward", 5.0, 2, vec![1, 1, 1, 0, 0, 0]),
        (WARD, "ward", 100.0, 1, vec![0, 0, 0, 0, 0, 0]),
        (COMPLETE, "complete", 0.1, 6, vec![5, 2, 3, 4, 1, 0]),
        (COMPLETE, "complete", 0.3, 5, vec![0, 0, 3, 4, 1, 2]),
        (COMPLETE, "complete", 1.5, 3, vec![0, 0, 0, 1, 1, 2]),
        (COMPLETE, "complete", 5.0, 2, vec![1, 1, 1, 0, 0, 0]),
        (COMPLETE, "complete", 100.0, 1, vec![0, 0, 0, 0, 0, 0]),
        (AVERAGE, "average", 0.1, 6, vec![5, 4, 3, 1, 2, 0]),
        (AVERAGE, "average", 0.3, 5, vec![0, 0, 3, 4, 2, 1]),
        (AVERAGE, "average", 1.5, 3, vec![0, 0, 0, 2, 2, 1]),
        (AVERAGE, "average", 5.0, 2, vec![0, 0, 0, 0, 0, 1]),
        (AVERAGE, "average", 100.0, 1, vec![0, 0, 0, 0, 0, 0]),
        (SINGLE, "single", 0.1, 6, vec![5, 3, 2, 4, 1, 0]),
        (SINGLE, "single", 0.3, 5, vec![0, 0, 2, 4, 3, 1]),
        (SINGLE, "single", 1.5, 3, vec![0, 0, 0, 2, 2, 1]),
        (SINGLE, "single", 5.0, 1, vec![0, 0, 0, 0, 0, 0]),
        (SINGLE, "single", 100.0, 1, vec![0, 0, 0, 0, 0, 0]),
    ];

    let x = fixture6();
    for (lk, name, t, exp_k, exp_labels) in cases {
        let fitted = AgglomerativeClustering::<f64>::new(2) // n_clusters cleared by with_distance_threshold
            .with_linkage(lk)
            .with_distance_threshold(t)
            .fit(&x, &())
            .unwrap();
        assert_eq!(
            fitted.n_clusters(),
            exp_k,
            "6pt {name} t={t}: n_clusters_ {} != oracle {exp_k}",
            fitted.n_clusters()
        );
        let labels: Vec<usize> = fitted.labels().iter().copied().collect();
        assert_eq!(
            labels, exp_labels,
            "6pt {name} t={t}: labels_ {labels:?} != oracle {exp_labels:?}"
        );
    }
}

/// REQ-9: the `distance_threshold` path on the 10-pt fixture, all four linkages.
///
/// Live sklearn 1.5.2 oracle (system python3):
/// ```text
/// python3 -c "import numpy as np; from sklearn.cluster import AgglomerativeClustering; \
///   X=np.array([[0.,0.],[0.2,0.1],[0.9,1.1],[3.,3.2],[3.3,3.],[6.,0.2],[6.1,0.5],[1.0,5.0],[1.2,5.3],[5.5,5.5]]); \
///   [print(L, t, m.n_clusters_, m.labels_.tolist()) \
///      for L in ['ward','complete','average','single'] for t in [0.25,1.0,4.0] \
///      for m in [AgglomerativeClustering(n_clusters=None,distance_threshold=t,linkage=L).fit(X)]]"
/// # ward     0.25 9 [0,0,5,6,7,8,4,3,2,1]   ward     1.0 6 [3,3,5,0,0,2,2,1,1,4]   ward     4.0 4 [1,1,1,0,0,2,2,3,3,0]
/// # complete 0.25 9 [0,0,5,6,4,8,7,3,2,1]   complete 1.0 6 [4,4,5,0,0,2,2,1,1,3]   complete 4.0 4 [1,1,1,0,0,2,2,0,0,3]
/// # average  0.25 9 [0,0,5,6,4,8,2,7,3,1]   average  1.0 6 [2,2,5,0,0,4,4,1,1,3]   average  4.0 3 [2,2,2,0,0,1,1,0,0,0]
/// # single   0.25 9 [0,0,4,6,7,5,8,3,1,2]   single   1.0 6 [2,2,4,0,0,3,3,1,1,5]   single   4.0 1 [0,0,0,0,0,0,0,0,0,0]
/// ```
#[test]
fn threshold_cut_exact_10pt_all_linkages() {
    type Case = (Linkage, &'static str, f64, usize, Vec<usize>);
    let cases: Vec<Case> = vec![
        (WARD, "ward", 0.25, 9, vec![0, 0, 5, 6, 7, 8, 4, 3, 2, 1]),
        (WARD, "ward", 1.0, 6, vec![3, 3, 5, 0, 0, 2, 2, 1, 1, 4]),
        (WARD, "ward", 4.0, 4, vec![1, 1, 1, 0, 0, 2, 2, 3, 3, 0]),
        (
            COMPLETE,
            "complete",
            0.25,
            9,
            vec![0, 0, 5, 6, 4, 8, 7, 3, 2, 1],
        ),
        (
            COMPLETE,
            "complete",
            1.0,
            6,
            vec![4, 4, 5, 0, 0, 2, 2, 1, 1, 3],
        ),
        (
            COMPLETE,
            "complete",
            4.0,
            4,
            vec![1, 1, 1, 0, 0, 2, 2, 0, 0, 3],
        ),
        (
            AVERAGE,
            "average",
            0.25,
            9,
            vec![0, 0, 5, 6, 4, 8, 2, 7, 3, 1],
        ),
        (
            AVERAGE,
            "average",
            1.0,
            6,
            vec![2, 2, 5, 0, 0, 4, 4, 1, 1, 3],
        ),
        (
            AVERAGE,
            "average",
            4.0,
            3,
            vec![2, 2, 2, 0, 0, 1, 1, 0, 0, 0],
        ),
        (
            SINGLE,
            "single",
            0.25,
            9,
            vec![0, 0, 4, 6, 7, 5, 8, 3, 1, 2],
        ),
        (SINGLE, "single", 1.0, 6, vec![2, 2, 4, 0, 0, 3, 3, 1, 1, 5]),
        (SINGLE, "single", 4.0, 1, vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    ];

    let x = fixture10();
    for (lk, name, t, exp_k, exp_labels) in cases {
        let fitted = AgglomerativeClustering::<f64>::new(2)
            .with_linkage(lk)
            .with_distance_threshold(t)
            .fit(&x, &())
            .unwrap();
        assert_eq!(
            fitted.n_clusters(),
            exp_k,
            "10pt {name} t={t}: n_clusters_ {} != oracle {exp_k}",
            fitted.n_clusters()
        );
        let labels: Vec<usize> = fitted.labels().iter().copied().collect();
        assert_eq!(
            labels, exp_labels,
            "10pt {name} t={t}: labels_ {labels:?} != oracle {exp_labels:?}"
        );
    }
}

/// REQ-9: the `n_clusters_ = count_nonzero(distances_ >= t) + 1` rule
/// (`_agglomerative.py:1090-1093`) holds for a threshold exactly equal to one of
/// the merge distances (`>=` is inclusive). Using the ward 6-pt distances_, the
/// 3rd merge distance is `1.5242484486898236`; a threshold equal to it counts
/// that merge (and the two above) as "not merged" → `3 + 1 = 4`? No: counts
/// merges with distance >= t. For t == d[2], merges {2,3,4} qualify → 3 + 1 = 4.
///
/// Live sklearn 1.5.2 oracle:
/// ```text
/// python3 -c "import numpy as np; from sklearn.cluster import AgglomerativeClustering; \
///   X=np.array([[0.,0.],[0.2,0.1],[0.9,1.1],[3.,3.2],[3.3,3.],[6.,0.2]]); \
///   t=1.5242484486898236; \
///   print(AgglomerativeClustering(n_clusters=None,distance_threshold=t,linkage='ward').fit(X).n_clusters_)"
/// # 4
/// ```
#[test]
fn threshold_inclusive_boundary_6pt_ward() {
    let x = fixture6();
    let t = 1.5242484486898236_f64;
    let fitted = AgglomerativeClustering::<f64>::new(2)
        .with_linkage(WARD)
        .with_distance_threshold(t)
        .fit(&x, &())
        .unwrap();
    assert_eq!(
        fitted.n_clusters(),
        4,
        "inclusive >= boundary: n_clusters_ should be 4"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// REQ-10 — n_leaves_ / n_connected_components_
// ─────────────────────────────────────────────────────────────────────────────

/// REQ-10: `n_leaves_ == n_samples` and `n_connected_components_ == 1` for the
/// unstructured (`connectivity=None`) path, all four linkages, both fixtures.
///
/// Live sklearn 1.5.2 oracle (system python3):
/// ```text
/// python3 -c "import numpy as np; from sklearn.cluster import AgglomerativeClustering; \
///   X=np.array([[0.,0.],[0.2,0.1],[0.9,1.1],[3.,3.2],[3.3,3.],[6.,0.2]]); \
///   m=AgglomerativeClustering(n_clusters=2).fit(X); print(m.n_leaves_, m.n_connected_components_)"
/// # 6 1
/// ```
#[test]
fn n_leaves_and_connected_components_unstructured() {
    for (x, n) in [(fixture6(), 6usize), (fixture10(), 10usize)] {
        for lk in [WARD, COMPLETE, AVERAGE, SINGLE] {
            let fitted = AgglomerativeClustering::<f64>::new(2)
                .with_linkage(lk)
                .fit(&x, &())
                .unwrap();
            assert_eq!(
                fitted.n_leaves(),
                n,
                "{lk:?}: n_leaves_ should equal n_samples"
            );
            assert_eq!(
                fitted.n_connected_components(),
                1,
                "{lk:?}: n_connected_components_ should be 1 (unstructured)"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// REQ-9 — XOR validation (n_clusters ^ distance_threshold)
// ─────────────────────────────────────────────────────────────────────────────

/// REQ-9: the XOR contract (`_agglomerative.py:1022-1027`). sklearn raises
/// `ValueError("Exactly one of n_clusters and distance_threshold has to be
/// set, and the other needs to be None.")` when neither or both is set.
///
/// Live sklearn 1.5.2 oracle (system python3):
/// ```text
/// python3 -c "import numpy as np; from sklearn.cluster import AgglomerativeClustering; \
///   X=np.array([[0.,0.],[0.2,0.1],[10.,10.],[10.1,10.]]); \
///   AgglomerativeClustering(n_clusters=2,distance_threshold=5.0).fit(X)"
/// # ValueError: Exactly one of n_clusters and distance_threshold has to be set, and the other needs to be None.
/// python3 -c "import numpy as np; from sklearn.cluster import AgglomerativeClustering; \
///   X=np.array([[0.,0.],[0.2,0.1],[10.,10.],[10.1,10.]]); \
///   AgglomerativeClustering(n_clusters=None).fit(X)"
/// # ValueError: Exactly one of n_clusters and distance_threshold has to be set, and the other needs to be None.
/// ```
#[test]
fn xor_both_set_errors() {
    let x = fixture6();
    // Both set: start from n_clusters=2 then ALSO set the field directly.
    let mut model = AgglomerativeClustering::<f64>::new(2);
    model.distance_threshold = Some(5.0); // n_clusters still Some(2) → both set
    let res = model.fit(&x, &());
    assert!(
        res.is_err(),
        "both n_clusters and distance_threshold set must error"
    );
}

/// REQ-9: neither set → error (mirrors sklearn `n_clusters=None,
/// distance_threshold=None`).
#[test]
fn xor_neither_set_errors() {
    let x = fixture6();
    let mut model = AgglomerativeClustering::<f64>::new(2);
    model.n_clusters = None; // both None now
    let res = model.fit(&x, &());
    assert!(
        res.is_err(),
        "neither n_clusters nor distance_threshold set must error"
    );
}

/// REQ-9: each single-set path fits successfully (the two legal XOR states).
#[test]
fn xor_single_set_each_fits() {
    let x = fixture6();

    // n_clusters only.
    let f1 = AgglomerativeClustering::<f64>::new(3)
        .with_linkage(WARD)
        .fit(&x, &());
    assert!(f1.is_ok(), "n_clusters-only must fit");
    assert_eq!(f1.unwrap().n_clusters(), 3);

    // distance_threshold only (n_clusters cleared by the builder).
    let f2 = AgglomerativeClustering::<f64>::new(2)
        .with_distance_threshold(5.0)
        .with_linkage(WARD)
        .fit(&x, &());
    assert!(f2.is_ok(), "distance_threshold-only must fit");
}
