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
//! 2. **REQ-2/REQ-4/REQ-8 value-parity guards (PASS today; #984/#986/#990).**
//!    Assert that `cluster_centers_` VALUES (~1e-6) and `labels_` INTEGERS
//!    (exact) match the live sklearn 1.5.2 `MeanShift(bandwidth=...).fit(X)`,
//!    not merely the partition. sklearn post-processing
//!    (`_mean_shift.py:529-557`): converged modes sorted by
//!    `(intensity, coord)` DESCENDING, greedy radius-`bandwidth` unique
//!    selection, labels = nearest retained center in that sorted order. The
//!    converged mode VALUES match because `mean_shift_single` uses
//!    `stop_thresh = 1e-3 * bandwidth` (REQ-8, `_mean_shift.py:113`).
//!
//! 3. **REQ-3 divergence pin (FAIL until fixed; tracking #995).** sklearn's
//!    auto-bandwidth heuristic `estimate_bandwidth(X, quantile=0.3)`
//!    (`_mean_shift.py:43-106`) is a kNN statistic; ferrolearn's private
//!    `fn estimate_bandwidth` computes the MEDIAN of all pairwise distances —
//!    a different value, so `MeanShift::new().fit(X)` (auto-bandwidth) yields a
//!    different `n_clusters` than `sklearn.cluster.MeanShift().fit(X)`.

use ferrolearn_cluster::MeanShift;
use ferrolearn_core::traits::Fit;
use ndarray::Array2;

/// Assert `cluster_centers_` matches `expected_centers` (row-for-row, ~1e-6)
/// and `labels_` matches `expected_labels` (exact integers) for an
/// explicit-bandwidth fit. Both expectations come from the live sklearn 1.5.2
/// oracle (R-CHAR-3).
fn assert_centers_and_labels(
    x: &Array2<f64>,
    bw: f64,
    expected_centers: &[[f64; 2]],
    expected_labels: &[usize],
) {
    let fitted = MeanShift::<f64>::new()
        .with_bandwidth(bw)
        .fit(x, &())
        .unwrap();
    let centers = fitted.cluster_centers();
    assert_eq!(
        centers.nrows(),
        expected_centers.len(),
        "n_clusters must match sklearn (bw={bw})"
    );
    for (i, exp) in expected_centers.iter().enumerate() {
        for (j, &e) in exp.iter().enumerate() {
            let got = centers[[i, j]];
            assert!(
                (got - e).abs() < 1e-6,
                "cluster_centers_[{i}][{j}] = {got} != sklearn {e} (bw={bw})"
            );
        }
    }
    assert_eq!(
        fitted.labels().as_slice().unwrap(),
        expected_labels,
        "labels_ integers must match sklearn exactly (bw={bw})"
    );
}

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
    let sklearn_partition: Vec<Vec<usize>> = vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8]];
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
/// Before #985 ferrolearn `fn estimate_bandwidth(X)` was the median pairwise
/// distance = 20.0004998750…, at which bandwidth even sklearn collapses to 1
/// cluster — so auto-`MeanShift` yielded `n_clusters() == 1`. After #985 the
/// estimator is sklearn's kNN heuristic (`quantile = 0.3`), recovering the
/// sklearn bandwidth and `n_clusters() == 3`.
///
/// sklearn expects 3; ferrolearn now matches (kNN `estimate_bandwidth`,
/// fixed in #985 / tracking #995).
/// Tracking: #995
#[test]
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

// ─────────────────────────────────────────────────────────────────────────────
// REQ-2 / REQ-4 / REQ-8 value-parity guards: cluster_centers_ VALUES (~1e-6)
// + labels_ INTEGERS (exact) match live sklearn 1.5.2 MeanShift(bandwidth=b).
// All expected values from the live oracle (R-CHAR-3):
//   m = MeanShift(bandwidth=b).fit(np.array(X, float))
//   print(m.cluster_centers_.tolist(), m.labels_.tolist())
// ─────────────────────────────────────────────────────────────────────────────

/// REQ-2/REQ-4: the upstream MeanShift docstring example
/// (`_mean_shift.py:428-434`), `bandwidth=2`.
///
/// Live sklearn 1.5.2:
/// ```text
/// X=[[1,1],[2,1],[1,0],[4,7],[3,5],[3,6]]
/// MeanShift(bandwidth=2).fit(X).cluster_centers_
///   -> [[3.3333333333333335, 6.0], [1.3333333333333333, 0.6666666666666666]]
/// .labels_ -> [1, 1, 1, 0, 0, 0]
/// ```
/// Both modes have intensity 3, so the `(intensity, coord)`-DESC tie is broken
/// by coordinate: `(3.33,6.0) > (1.33,0.66)` → it sorts first (index 0), which
/// is why the first three points (closest to `(1.33,0.66)`) get label 1.
#[test]
fn green_docstring_centers_and_labels_match_sklearn_bw2() {
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![1.0, 1.0, 2.0, 1.0, 1.0, 0.0, 4.0, 7.0, 3.0, 5.0, 3.0, 6.0],
    )
    .unwrap();
    let expected_centers = [
        [3.333_333_333_333_333_5, 6.0],
        [1.333_333_333_333_333_3, 0.666_666_666_666_666_6],
    ];
    let expected_labels = [1, 1, 1, 0, 0, 0];
    assert_centers_and_labels(&x, 2.0, &expected_centers, &expected_labels);
}

/// REQ-2/REQ-4/REQ-8: two well-separated blobs, `bandwidth=2.0` (`!= 1`, so the
/// `stop_thresh = 1e-3 * bandwidth` scaling is exercised — REQ-8).
///
/// Live sklearn 1.5.2:
/// ```text
/// X=[[0,0],[0.2,0.1],[-0.1,0.2],[0.1,-0.1],[0,0.1],
///    [10,10],[10.2,10.1],[9.9,10.2],[10.1,9.9],[10,10.1]]
/// MeanShift(bandwidth=2.0).fit(X).cluster_centers_
///   -> [[10.040000000000001, 10.06], [0.04, 0.06000000000000001]]
/// .labels_ -> [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
/// ```
/// The `(10,10)` mode (5 points) and the origin mode (5 points) tie on
/// intensity, so the coordinate-DESC tie puts `(10.04,10.06)` first (index 0)
/// and the origin points get label 1.
#[test]
fn green_two_blobs_centers_and_labels_match_sklearn_bw2() {
    let x = Array2::from_shape_vec(
        (10, 2),
        vec![
            0.0, 0.0, 0.2, 0.1, -0.1, 0.2, 0.1, -0.1, 0.0, 0.1, 10.0, 10.0, 10.2, 10.1, 9.9, 10.2,
            10.1, 9.9, 10.0, 10.1,
        ],
    )
    .unwrap();
    let expected_centers = [
        [10.040_000_000_000_001, 10.06],
        [0.04, 0.060_000_000_000_000_01],
    ];
    let expected_labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
    assert_centers_and_labels(&x, 2.0, &expected_centers, &expected_labels);
}

/// REQ-2/REQ-4/REQ-8: three blobs (origin / (10,0) / (0,10)), `bandwidth=1.5`.
///
/// Live sklearn 1.5.2:
/// ```text
/// X=[[0,0],[0.1,0],[0,0.1],[10,0],[10.1,0],[10,0.1],[0,10],[0.1,10],[0,10.1]]
/// MeanShift(bandwidth=1.5).fit(X).cluster_centers_
///   -> [[10.033333333333333, 0.03333333333333333],
///       [0.03333333333333333, 10.033333333333333],
///       [0.03333333333333333, 0.03333333333333333]]
/// .labels_ -> [2, 2, 2, 0, 0, 0, 1, 1, 1]
/// ```
/// All three modes have intensity 3; the coordinate-DESC order is
/// `(10.03,0.03) > (0.03,10.03) > (0.03,0.03)`, so the origin block is the LAST
/// center (label 2).
#[test]
fn green_three_blobs_centers_and_labels_match_sklearn_bw15() {
    let x = Array2::from_shape_vec(
        (9, 2),
        vec![
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 10.0, 0.0, 10.1, 0.0, 10.0, 0.1, 0.0, 10.0, 0.1, 10.0,
            0.0, 10.1,
        ],
    )
    .unwrap();
    let expected_centers = [
        [10.033_333_333_333_333, 0.033_333_333_333_333_33],
        [0.033_333_333_333_333_33, 10.033_333_333_333_333],
        [0.033_333_333_333_333_33, 0.033_333_333_333_333_33],
    ];
    let expected_labels = [2, 2, 2, 0, 0, 0, 1, 1, 1];
    assert_centers_and_labels(&x, 1.5, &expected_centers, &expected_labels);
}

/// REQ-2/REQ-4/REQ-8: a larger bandwidth (`bandwidth=5.0`) over two clusters at
/// `(1,1)` and `(8,8)`, exercising the `1e-3 * bandwidth` stop-threshold (REQ-8)
/// far from `bandwidth == 1`.
///
/// Live sklearn 1.5.2:
/// ```text
/// X=[[1,1],[1.5,0.8],[0.7,1.2],[8,8],[8.4,7.6],[7.6,8.3]]
/// MeanShift(bandwidth=5.0).fit(X).cluster_centers_
///   -> [[8.0, 7.966666666666666], [1.0666666666666667, 1.0]]
/// .labels_ -> [1, 1, 1, 0, 0, 0]
/// ```
#[test]
fn green_two_clusters_centers_and_labels_match_sklearn_bw5() {
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![1.0, 1.0, 1.5, 0.8, 0.7, 1.2, 8.0, 8.0, 8.4, 7.6, 7.6, 8.3],
    )
    .unwrap();
    let expected_centers = [[8.0, 7.966_666_666_666_666], [1.066_666_666_666_666_7, 1.0]];
    let expected_labels = [1, 1, 1, 0, 0, 0];
    assert_centers_and_labels(&x, 5.0, &expected_centers, &expected_labels);
}

/// REQ-9 side-effect of the REQ-8 fix: `n_iter_` now matches sklearn's
/// `max([completed_iterations])` convention (`_mean_shift.py:124-129,514`)
/// because `mean_shift_single` mirrors `_mean_shift_single_seed` exactly
/// (increment AFTER the convergence/`max_iter` check, return
/// `completed_iterations`).
///
/// Live sklearn 1.5.2: `MeanShift(bandwidth=2).fit(docs).n_iter_ == 2`.
#[test]
fn green_n_iter_matches_sklearn_docs_bw2() {
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![1.0, 1.0, 2.0, 1.0, 1.0, 0.0, 4.0, 7.0, 3.0, 5.0, 3.0, 6.0],
    )
    .unwrap();
    // Live sklearn 1.5.2 oracle: n_iter_ == 2.
    const SKLEARN_N_ITER: usize = 2;
    let fitted = MeanShift::<f64>::new()
        .with_bandwidth(2.0)
        .fit(&x, &())
        .unwrap();
    assert_eq!(
        fitted.n_iter(),
        SKLEARN_N_ITER,
        "n_iter_ must match sklearn's max(completed_iterations) convention"
    );
}
