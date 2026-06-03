//! Divergence / green-guard tests for `ferrolearn_cluster::Hdbscan` against the
//! live scikit-learn 1.5.2 HDBSCAN oracle.
//!
//! Upstream: `sklearn/cluster/_hdbscan/hdbscan.py`
//! (`class HDBSCAN(ClusterMixin, BaseEstimator)`).
//!
//! All EXPECTED values were produced by a live `from sklearn.cluster import
//! HDBSCAN` call run from `/tmp` (sklearn 1.5.2 == the source clone at tag
//! 1.5.2, commit 156ef14) and are recorded inline as `SK_*` constants. They are
//! NEVER copied from the ferrolearn side (R-CHAR-3). HDBSCAN is fully
//! deterministic (no RNG), so the partition is a pure function of the data and
//! value-parity of the partition is genuinely checkable.
//!
//! Tests fall into two groups:
//!   * Green-guards (PASS today): REQ-1 partition + noise, REQ-2
//!     probability range + noise-prob-0, REQ-3 defaults. These MUST stay green
//!     after the REQ-5 core-distance fix — their fixtures are well-separated and
//!     give the SAME sklearn partition at `min_samples=k` and `k+1`.
//!   * REQ-5 pin (FAILS today, green after the one-line core-distance fix):
//!     `divergence_core_distance_off_by_one_partition`.

use ferrolearn_cluster::Hdbscan;
use ferrolearn_core::Fit;
use ndarray::Array2;

/// Canonicalize a label vector into (sorted groups of indices, sorted noise
/// indices). Two clusterings are equal up-to-permutation iff their canonical
/// forms are equal. Noise (`-1`) is tracked as its own index set.
fn canon(labels: &[isize]) -> (Vec<Vec<usize>>, Vec<usize>) {
    use std::collections::BTreeMap;
    let mut noise = Vec::new();
    let mut groups: BTreeMap<isize, Vec<usize>> = BTreeMap::new();
    for (i, &l) in labels.iter().enumerate() {
        if l == -1 {
            noise.push(i);
        } else {
            groups.entry(l).or_default().push(i);
        }
    }
    let mut g: Vec<Vec<usize>> = groups.into_values().collect();
    for grp in &mut g {
        grp.sort_unstable();
    }
    g.sort();
    noise.sort_unstable();
    (g, noise)
}

fn fit_canon(x: &Array2<f64>, mcs: usize, ms: usize) -> (Vec<Vec<usize>>, Vec<usize>) {
    let fitted = Hdbscan::<f64>::new()
        .with_min_cluster_size(mcs)
        .with_min_samples(ms)
        .fit(x, &())
        .expect("fit must succeed");
    canon(fitted.labels().as_slice().expect("contiguous labels"))
}

// ───────────────────────────── fixtures ─────────────────────────────

/// Two well-separated 6-point blobs near (0,0) and (10,10). 12 points.
fn well_separated_two_blobs() -> Array2<f64> {
    Array2::from_shape_vec(
        (12, 2),
        vec![
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1, 0.05, 0.05, -0.05, 0.05, 10.0, 10.0, 10.1,
            10.0, 10.0, 10.1, 10.1, 10.1, 10.05, 10.05, 9.95, 10.05,
        ],
    )
    .unwrap()
}

/// Two 5-point blobs near (0,0) and (10,10) plus four far outliers. 14 points.
fn two_blobs_with_outliers() -> Array2<f64> {
    Array2::from_shape_vec(
        (14, 2),
        vec![
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1, 0.05, 0.05, // blob A (0..4)
            10.0, 10.0, 10.1, 10.0, 10.0, 10.1, 10.1, 10.1, 10.05, 10.05, // blob B (5..9)
            50.0, 50.0, -50.0, -50.0, 100.0, 0.0, 0.0, 100.0, // outliers (10..13)
        ],
    )
    .unwrap()
}

// ───────────────────── REQ-1: partition + noise (GREEN GUARD) ─────────────────────

/// Green-guard for REQ-1 (SHIPPED): on the well-separated two-blob fixture,
/// `HDBSCAN(min_cluster_size=3).fit(X).labels_` partitions as {0..5} / {6..11}
/// with no noise. ferrolearn's partition (co-membership up-to-permutation) +
/// noise set must match the live oracle.
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.cluster import HDBSCAN; \
///   X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[0.1,0.1],[0.05,0.05],[-0.05,0.05], \
///   [10.,10.],[10.1,10.],[10.,10.1],[10.1,10.1],[10.05,10.05],[9.95,10.05]]); \
///   print(HDBSCAN(min_cluster_size=3).fit(X).labels_.tolist())"
///   -> [0,0,0,0,0,0,1,1,1,1,1,1]
/// Robust to the REQ-5 fix: sklearn gives this SAME partition at min_samples=3
/// AND min_samples=4 (verified live), so the guard stays green after the fix.
#[test]
fn req1_partition_two_blobs_matches_sklearn() {
    // sklearn labels [0,0,0,0,0,0,1,1,1,1,1,1] -> canonical partition.
    let sk_groups: Vec<Vec<usize>> = vec![vec![0, 1, 2, 3, 4, 5], vec![6, 7, 8, 9, 10, 11]];
    let sk_noise: Vec<usize> = vec![];

    let x = well_separated_two_blobs();
    // min_samples=3 (= min_cluster_size). sklearn's partition is identical at
    // min_samples=3 and 4, so this stays green after the REQ-5 core-distance fix.
    let (g, n) = fit_canon(&x, 3, 3);
    assert_eq!(g, sk_groups, "partition must match sklearn (two blobs)");
    assert_eq!(n, sk_noise, "no noise expected (two blobs)");
}

/// Green-guard for REQ-1 (SHIPPED): outlier fixture. sklearn labels the four
/// scattered outliers `-1` and clusters the two blobs.
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.cluster import HDBSCAN; \
///   X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[0.1,0.1],[0.05,0.05], \
///   [10.,10.],[10.1,10.],[10.,10.1],[10.1,10.1],[10.05,10.05], \
///   [50.,50.],[-50.,-50.],[100.,0.],[0.,100.]]); \
///   print(HDBSCAN(min_cluster_size=3).fit(X).labels_.tolist())"
///   -> [0,0,0,0,0,1,1,1,1,1,-1,-1,-1,-1]
/// Robust to REQ-5: sklearn gives this SAME partition + noise set at
/// min_samples=3 AND min_samples=4 (verified live).
#[test]
fn req1_partition_outliers_matches_sklearn() {
    // sklearn labels [0,0,0,0,0,1,1,1,1,1,-1,-1,-1,-1].
    let sk_groups: Vec<Vec<usize>> = vec![vec![0, 1, 2, 3, 4], vec![5, 6, 7, 8, 9]];
    let sk_noise: Vec<usize> = vec![10, 11, 12, 13];

    let x = two_blobs_with_outliers();
    let (g, n) = fit_canon(&x, 3, 3);
    assert_eq!(
        g, sk_groups,
        "partition must match sklearn (outlier fixture)"
    );
    assert_eq!(
        n, sk_noise,
        "the four scattered outliers must be noise (-1)"
    );
}

// ───────────────── REQ-2: probabilities range + noise-prob-0 (GREEN GUARD) ─────────────────

/// Green-guard for REQ-2 (SHIPPED): every probability is in [0,1] and every
/// noise point has probability exactly 0. This is the CONTRACT only — the exact
/// GLOSH values diverge (REQ-4 / #1069) and are NOT asserted here.
///
/// Oracle (sklearn 1.5.2, run from /tmp) on the outlier fixture:
///   ... print([round(p,3) for p in HDBSCAN(min_cluster_size=3).fit(X).probabilities_])
///   -> [1,1,1,1,1, 1,1,1,1,1, 0,0,0,0]   (all four outliers prob 0)
#[test]
fn req2_probabilities_range_and_noise_zero() {
    let x = two_blobs_with_outliers();
    let fitted = Hdbscan::<f64>::new()
        .with_min_cluster_size(3)
        .with_min_samples(3)
        .fit(&x, &())
        .unwrap();

    let labels = fitted.labels();
    let probs = fitted.probabilities();
    assert_eq!(labels.len(), probs.len());

    for (i, &p) in probs.iter().enumerate() {
        assert!(
            (0.0..=1.0).contains(&p),
            "probability[{i}] = {p} out of [0,1]"
        );
    }
    // sklearn: noise points have probability exactly 0 (hdbscan.py:556-562).
    for (i, &l) in labels.iter().enumerate() {
        if l == -1 {
            assert_eq!(probs[i], 0.0, "noise point {i} must have probability 0");
        }
    }
}

// ───────────────────────── REQ-3: defaults (GREEN GUARD) ─────────────────────────

/// Green-guard for REQ-3 (SHIPPED): constructor defaults mirror sklearn's
/// `__init__` (`hdbscan.py:674-676`): `min_cluster_size=5`, `min_samples=None`,
/// `cluster_selection_epsilon=0.0`.
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "from sklearn.cluster import HDBSCAN; h=HDBSCAN(); \
///   print(h.min_cluster_size, h.min_samples, h.cluster_selection_epsilon)"
///   -> 5 None 0.0
#[test]
fn req3_defaults_match_sklearn() {
    // sklearn HDBSCAN() defaults (hdbscan.py:674-676).
    const SK_MIN_CLUSTER_SIZE: usize = 5;
    const SK_CLUSTER_SELECTION_EPSILON: f64 = 0.0;
    // sklearn min_samples default is None.

    let m = Hdbscan::<f64>::new();
    assert_eq!(m.min_cluster_size, SK_MIN_CLUSTER_SIZE);
    assert!(m.min_samples.is_none(), "min_samples default must be None");
    assert_eq!(m.cluster_selection_epsilon, SK_CLUSTER_SELECTION_EPSILON);
}

// ───────── REQ-5 PIN: core-distance off-by-one (FAILS today, green after fix) ─────────

/// Divergence pin for REQ-5 (#1070): ferrolearn's `fn compute_core_distances`
/// takes `dists[min_samples]` (== sklearn `sorted_dists[min_samples]`), but
/// sklearn's core distance is `neighbors_distances[:, -1]` of the `min_samples`
/// nearest neighbors INCLUDING self == `sorted_dists[min_samples - 1]`
/// (`sklearn/cluster/_hdbscan/hdbscan.py:351-352`,
///  `core_distances = np.ascontiguousarray(neighbors_distances[:, -1])`).
/// So ferrolearn at `min_samples=k` reproduces sklearn at `min_samples=k+1`.
///
/// This is observable in the PARTITION on a min_samples-sensitive fixture.
///
/// Fixture: a tight 5-point blob, a 5-point linear chain with growing gaps, and
/// a second tight 5-point blob. The chain's density boundary makes the partition
/// min_samples-sensitive.
///
/// Oracle (sklearn 1.5.2, run from /tmp), min_cluster_size=3:
///   X = [[0,0],[0.1,0.1],[-0.1,0.05],[0.05,-0.1],[0,0.15],
///        [1,0],[1.5,0],[2,0],[2.5,0],[3,0],
///        [5,5],[5.1,5.1],[4.9,5.05],[5.05,4.9],[5,5.15]]
///   ms=3 -> groups [[0,1,2,3,4,5],[6,7,8],[10,11,12,13,14]] noise=[9]
///   ms=4 -> groups [[0..9],[10..14]] noise=[]
/// The two partitions DIFFER (the fixture is min_samples-sensitive at k=3).
///
/// ferrolearn at min_samples=3 (off-by-one) currently reproduces sklearn's
/// min_samples=4 partition, so it FAILS the "matches sklearn at min_samples=3"
/// assertion below. After the one-line fix (`dists[min_samples]` ->
/// `dists[min_samples - 1]`), ferrolearn at min_samples=3 will use sklearn's
/// min_samples=3 core distances and produce sklearn's min_samples=3 partition,
/// turning this test green. (Verified live: ferrolearn current ms=k == sklearn
/// ms=(k+1) EXACTLY across k=2..5 on this fixture, isolating REQ-5 from the
/// probability/label-integer divergences.)
///
/// Tracking: #1070
#[test]
fn divergence_core_distance_off_by_one_partition() {
    // sklearn HDBSCAN(min_cluster_size=3, min_samples=3) partition on the fixture.
    let sk_groups_ms3: Vec<Vec<usize>> = vec![
        vec![0, 1, 2, 3, 4, 5],
        vec![6, 7, 8],
        vec![10, 11, 12, 13, 14],
    ];
    let sk_noise_ms3: Vec<usize> = vec![9];

    let x = Array2::from_shape_vec(
        (15, 2),
        vec![
            0.0, 0.0, 0.1, 0.1, -0.1, 0.05, 0.05, -0.1, 0.0, 0.15, // tight blob (0..4)
            1.0, 0.0, 1.5, 0.0, 2.0, 0.0, 2.5, 0.0, 3.0, 0.0, // chain (5..9)
            5.0, 5.0, 5.1, 5.1, 4.9, 5.05, 5.05, 4.9, 5.0, 5.15, // tight blob (10..14)
        ],
    )
    .unwrap();

    // ferrolearn at min_samples=3 must match sklearn at min_samples=3.
    let (g, n) = fit_canon(&x, 3, 3);
    assert_eq!(
        g, sk_groups_ms3,
        "ferrolearn min_samples=3 partition must match sklearn min_samples=3 \
         (off-by-one core distance, #1070)"
    );
    assert_eq!(
        n, sk_noise_ms3,
        "ferrolearn min_samples=3 noise set must match sklearn min_samples=3 (#1070)"
    );
}
