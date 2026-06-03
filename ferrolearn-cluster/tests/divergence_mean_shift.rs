//! Divergence tests for `ferrolearn-cluster` `MeanShift` against the live
//! scikit-learn 1.5.2 oracle (`sklearn/cluster/_mean_shift.py`).
//!
//! Two kinds of tests live here:
//!
//! 1. **Green-guards (PASS today).** Pin the SHIPPED in-regime contract —
//!    REQ-1: the explicit-bandwidth `labels_` PARTITION (label co-membership,
//!    up to permutation) on well-separated data. The expected partitions are
//!    computed by the live sklearn 1.5.2 oracle (R-CHAR-3) and canonicalized to
//!    a set-of-sorted-index-groups so label-integer permutation and the
//!    intensity-ordering divergence (REQ-2/REQ-4) do not affect the assertion.
//!
//! 2. **REQ-3 divergence pin (FAIL until fixed; tracking #995).** sklearn's
//!    auto-bandwidth heuristic `estimate_bandwidth(X, quantile=0.3)`
//!    (`_mean_shift.py:43-106`) is a kNN statistic; ferrolearn's private
//!    `fn estimate_bandwidth` computes the MEDIAN of all pairwise distances —
//!    a different value, so `MeanShift::new().fit(X)` (auto-bandwidth) yields a
//!    different `n_clusters` than `sklearn.cluster.MeanShift().fit(X)`.

use ferrolearn_cluster::MeanShift;
use ferrolearn_core::traits::Fit;
use ndarray::Array2;

/// Canonicalize a label vector to a set of sorted index-groups, ignoring the
/// integer label values (and thus any permutation / intensity-ordering).
fn canonical_partition(labels: &[usize]) -> Vec<Vec<usize>> {
    use std::collections::BTreeMap;
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, &l) in labels.iter().enumerate() {
        groups.entry(l).or_default().push(i);
    }
    let mut out: Vec<Vec<usize>> = groups.into_values().collect();
    for g in &mut out {
        g.sort_unstable();
    }
    out.sort();
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Green-guards: REQ-1 explicit-bandwidth PARTITION (PASS today).
// Expected partitions from live sklearn 1.5.2:
//   from collections import defaultdict; canon(MeanShift(bandwidth=b).fit(X).labels_)
// ─────────────────────────────────────────────────────────────────────────────

/// REQ-1 green-guard. `two_blobs` fixture (5 near origin + 5 near (10,10)),
/// explicit `bandwidth=2.0`.
///
/// Live sklearn 1.5.2:
/// ```text
/// X=np.array([[0.,0.],[0.2,0.1],[-0.1,0.2],[0.1,-0.1],[0.,0.1],
///             [10.,10.],[10.2,10.1],[9.9,10.2],[10.1,9.9],[10.,10.1]])
/// MeanShift(bandwidth=2.0).fit(X).labels_ -> [1,1,1,1,1,0,0,0,0,0]
/// canon -> [[0,1,2,3,4],[5,6,7,8,9]]
/// ```
#[test]
fn green_two_blobs_partition_matches_sklearn_bw2() {
    let x = Array2::from_shape_vec(
        (10, 2),
        vec![
            0.0, 0.0, 0.2, 0.1, -0.1, 0.2, 0.1, -0.1, 0.0, 0.1, 10.0, 10.0, 10.2, 10.1, 9.9, 10.2,
            10.1, 9.9, 10.0, 10.1,
        ],
    )
    .unwrap();
    let fitted = MeanShift::<f64>::new()
        .with_bandwidth(2.0)
        .fit(&x, &())
        .unwrap();
    // Expected from live sklearn 1.5.2 oracle (canonicalized).
    let sklearn_partition: Vec<Vec<usize>> = vec![vec![0, 1, 2, 3, 4], vec![5, 6, 7, 8, 9]];
    assert_eq!(
        canonical_partition(fitted.labels().as_slice().unwrap()),
        sklearn_partition,
        "explicit-bandwidth partition must match sklearn MeanShift(bandwidth=2.0)"
    );
}

/// REQ-1 green-guard. 3-blob fixture (origin / (10,0) / (0,10)),
/// explicit `bandwidth=1.5`.
///
/// Live sklearn 1.5.2:
/// ```text
/// X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[10.,0.],[10.1,0.],[10.,0.1],
///             [0.,10.],[0.1,10.],[0.,10.1]])
/// MeanShift(bandwidth=1.5).fit(X).labels_ -> [2,2,2,0,0,0,1,1,1]
/// canon -> [[0,1,2],[3,4,5],[6,7,8]]
/// ```
#[test]
fn green_three_blobs_partition_matches_sklearn_bw15() {
    let x = Array2::from_shape_vec(
        (9, 2),
        vec![
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 10.0, 0.0, 10.1, 0.0, 10.0, 0.1, 0.0, 10.0, 0.1, 10.0,
            0.0, 10.1,
        ],
    )
    .unwrap();
    let fitted = MeanShift::<f64>::new()
        .with_bandwidth(1.5)
        .fit(&x, &())
        .unwrap();
    let sklearn_partition: Vec<Vec<usize>> =
        vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8]];
    assert_eq!(
        canonical_partition(fitted.labels().as_slice().unwrap()),
        sklearn_partition,
        "explicit-bandwidth partition must match sklearn MeanShift(bandwidth=1.5)"
    );
}

/// REQ-1 green-guard on a FRESH separable fixture (NOT used by any in-tree
/// test): three blobs at (0,0)/(20,0)/(0,20), explicit `bandwidth=3.0`.
/// Confirms the partition-up-to-permutation contract holds beyond the
/// fixtures hard-coded into `mean_shift.rs`.
///
/// Live sklearn 1.5.2:
/// ```text
/// X=np.array([[0.,0.],[0.3,0.2],[-0.2,0.1],[0.1,-0.3],
///             [20.,0.],[20.3,0.2],[19.8,-0.1],[20.1,0.3],
///             [0.,20.],[0.2,20.3],[-0.1,19.8],[0.3,20.1]])
/// MeanShift(bandwidth=3.0).fit(X).labels_ -> [2,2,2,2,0,0,0,0,1,1,1,1]
/// canon -> [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
/// ```
#[test]
fn green_fresh_three_blobs_partition_matches_sklearn_bw3() {
    let x = Array2::from_shape_vec(
        (12, 2),
        vec![
            0.0, 0.0, 0.3, 0.2, -0.2, 0.1, 0.1, -0.3, 20.0, 0.0, 20.3, 0.2, 19.8, -0.1, 20.1, 0.3,
            0.0, 20.0, 0.2, 20.3, -0.1, 19.8, 0.3, 20.1,
        ],
    )
    .unwrap();
    let fitted = MeanShift::<f64>::new()
        .with_bandwidth(3.0)
        .fit(&x, &())
        .unwrap();
    let sklearn_partition: Vec<Vec<usize>> =
        vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7], vec![8, 9, 10, 11]];
    assert_eq!(
        canonical_partition(fitted.labels().as_slice().unwrap()),
        sklearn_partition,
        "explicit-bandwidth partition must match sklearn MeanShift(bandwidth=3.0) on a fresh fixture"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// REQ-3 divergence pin (FAIL until fixed; tracking #995).
// ─────────────────────────────────────────────────────────────────────────────

/// Divergence: ferrolearn's auto-bandwidth `fn estimate_bandwidth`
/// (median of all pairwise distances) diverges from
/// `sklearn/cluster/_mean_shift.py:43-106` (`estimate_bandwidth`, the kNN
/// heuristic `sum(max(kNN_dist,axis=1))/n` with `k=int(n*quantile)`,
/// `quantile=0.3`).
///
/// Observable: `MeanShift::new().fit(X)` (auto-bandwidth, no explicit bw) on a
/// well-separated 3-blob fixture (blobs at (0,0)/(20,0)/(0,20)).
///
/// Live sklearn 1.5.2:
/// ```text
/// X = the fresh 3-blob fixture below
/// estimate_bandwidth(X)            -> 0.4010125863780225  (quantile=0.3)
/// MeanShift().fit(X).cluster_centers_.shape[0] -> 3
/// ```
/// ferrolearn `fn estimate_bandwidth(X)` = median pairwise = 20.0004998750…,
/// at which bandwidth even sklearn collapses to 1 cluster — so ferrolearn's
/// auto-`MeanShift` yields `n_clusters() == 1`, not 3.
///
/// sklearn expects 3; ferrolearn returns 1.
/// Tracking: #995
#[test]
#[ignore = "divergence: estimate_bandwidth is median-pairwise not the kNN heuristic; tracking #995"]
fn divergence_auto_bandwidth_n_clusters() {
    let x = Array2::from_shape_vec(
        (12, 2),
        vec![
            0.0, 0.0, 0.3, 0.2, -0.2, 0.1, 0.1, -0.3, 20.0, 0.0, 20.3, 0.2, 19.8, -0.1, 20.1, 0.3,
            0.0, 20.0, 0.2, 20.3, -0.1, 19.8, 0.3, 20.1,
        ],
    )
    .unwrap();
    // Live sklearn 1.5.2: MeanShift().fit(X) auto-bandwidth -> 3 clusters.
    const SKLEARN_AUTO_N_CLUSTERS: usize = 3;
    let fitted = MeanShift::<f64>::new().fit(&x, &()).unwrap();
    assert_eq!(
        fitted.n_clusters(),
        SKLEARN_AUTO_N_CLUSTERS,
        "auto-bandwidth MeanShift must match sklearn's kNN estimate_bandwidth \
         cluster count; ferrolearn's median-pairwise bandwidth collapses to 1 cluster"
    );
}
