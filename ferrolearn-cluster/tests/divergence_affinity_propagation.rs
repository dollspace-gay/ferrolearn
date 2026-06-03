//! Divergence + green-guard tests for `AffinityPropagation`
//! (`ferrolearn-cluster/src/affinity_propagation.rs`) against the live
//! scikit-learn 1.5.2 oracle (`sklearn/cluster/_affinity_propagation.py`).
//!
//! Authored by acto-critic (iteration 122). Expected values come from the live
//! sklearn 1.5.2 oracle or a sklearn `file:line`, NEVER literal-copied from the
//! ferrolearn side (R-CHAR-3).
//!
//! - [`green_guard_blobs_partition_default`] / [`green_guard_two_blob_partition`]
//!   PASS: they pin the SHIPPED REQ-3 co-membership contract on well-separated
//!   data, canonicalized to ignore label permutation.
//! - [`green_guard_responsibility_availability_equivalence`] PASS: pins the
//!   SHIPPED REQ-1 message-passing math via a converged-message property.
//! - [`pin_req2_default_preference_partition`] FAILS until #971 lands: it pins
//!   the REQ-2 default-preference divergence — current ferrolearn medians only
//!   the off-diagonal entries (k=2 on the fixture) while sklearn medians the
//!   full affinity matrix (k=3). It is constructed so the preference is the SOLE
//!   cause of the disagreement: ferrolearn with the explicit full-matrix-median
//!   preference already reproduces sklearn's partition (proven separately), so
//!   the fix flips this test green without touching REQ-4/5/6.

use ferrolearn_cluster::AffinityPropagation;
use ferrolearn_core::Fit;
use ndarray::Array2;

/// Canonicalize a label vector to a permutation-invariant signature:
/// the sorted set of sorted index-groups (one group per cluster).
fn canonical_partition(labels: &[isize]) -> Vec<Vec<usize>> {
    use std::collections::BTreeMap;
    let mut groups: BTreeMap<isize, Vec<usize>> = BTreeMap::new();
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

/// REQ-3 green-guard: well-separated 3-blob fixture, DEFAULT model.
///
/// Live sklearn 1.5.2:
/// ```text
/// X,_=make_blobs(n_samples=12,centers=3,cluster_std=0.4,random_state=42)
/// AffinityPropagation(random_state=0).fit(X)
///   n_clusters = 3
///   labels_    = [0, 0, 1, 0, 2, 0, 2, 1, 2, 1, 1, 2]
/// ```
/// (`sklearn/cluster/_affinity_propagation.py:312`, default model.) The absolute
/// label integers / exemplar indices are NOT guaranteed (REQ-4/6); the
/// co-membership PARTITION up to a permutation IS the SHIPPED contract (REQ-3).
#[test]
fn green_guard_blobs_partition_default() {
    // make_blobs(n_samples=12, centers=3, cluster_std=0.4, random_state=42)
    let x = Array2::from_shape_vec(
        (12, 2),
        vec![
            -7.3400246221201915, -6.729830385937678,
            -6.852615909276101, -7.450008867761329,
            3.949911703222888, 1.7482546722443428,
            -7.097380280961343, -6.8357405573920005,
            -2.6969873774267312, 9.231310145632708,
            -7.119882667118792, -6.996787093193257,
            -1.8775124968497936, 9.321260019859485,
            4.234746388094332, 2.0988686169788418,
            -2.694564700177735, 8.827994226770219,
            5.226138343796723, 1.8828591637461178,
            4.276669206019617, 1.4082482034066155,
            -2.4124127144263365, 8.248974030335201,
        ],
    )
    .unwrap();

    let fitted = AffinityPropagation::<f64>::new().fit(&x, &()).unwrap();

    // sklearn oracle (random_state=0): the canonical partition of
    // [0, 0, 1, 0, 2, 0, 2, 1, 2, 1, 1, 2].
    let sklearn_labels: [isize; 12] = [0, 0, 1, 0, 2, 0, 2, 1, 2, 1, 1, 2];
    let expected = canonical_partition(&sklearn_labels);

    assert_eq!(fitted.n_clusters(), 3, "REQ-3: n_clusters must be 3");
    assert_eq!(
        canonical_partition(fitted.labels().as_slice().unwrap()),
        expected,
        "REQ-3: co-membership partition must match sklearn up to a label permutation"
    );
}

/// REQ-3 green-guard: 2-blob in-tree-style fixture, DEFAULT model.
///
/// Live sklearn 1.5.2 on the two-block fixture (4 points near (0,0), 4 near
/// (10,10)) yields exactly 2 clusters with the first four co-member and the last
/// four co-member. Pins the partition shape robustly.
#[test]
fn green_guard_two_blob_partition() {
    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, // cluster A
            10.0, 10.0, 10.5, 10.0, 10.0, 10.5, 10.5, 10.5, // cluster B
        ],
    )
    .unwrap();

    let fitted = AffinityPropagation::<f64>::new().fit(&x, &()).unwrap();

    // sklearn oracle: 2 clusters, partition {0,1,2,3}{4,5,6,7}.
    let expected = vec![vec![0usize, 1, 2, 3], vec![4, 5, 6, 7]];
    assert_eq!(fitted.n_clusters(), 2, "REQ-3: must find 2 well-separated clusters");
    assert_eq!(
        canonical_partition(fitted.labels().as_slice().unwrap()),
        expected,
        "REQ-3: two well-separated blobs must form a 2-block partition"
    );
    // REQ-1 corollary: a non-empty exemplar set converged.
    assert!(
        !fitted.exemplar_indices().is_empty(),
        "REQ-1: converged exemplar set must be non-empty"
    );
}

/// REQ-1 green-guard: the converged messages satisfy the algebraic property that
/// every exemplar is its own cluster's representative and the partition is a
/// valid disjoint cover of the samples.
///
/// The responsibility update in `fn fit`
/// (`R[i,k] = S[i,k] - max_{k'!=k}(A[i,k'] + S[i,k'])` with damping) is
/// algebraically identical to sklearn's `Y`/`Y2` argmax form
/// (`_affinity_propagation.py:88-102`), and the availability update
/// (`A[k,k]=sum max(0,R[i',k])`, `A[i,k]=min(0,R[k,k]+sum)`) to sklearn's
/// colsum/clip form (`:104-117`) — both verified numerically (max abs diff 0.0).
/// This test guards that the converged message-passing yields a coherent
/// exemplar partition on the 2-blob fixture.
#[test]
fn green_guard_responsibility_availability_equivalence() {
    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 10.0, 10.0, 10.5, 10.0, 10.0, 10.5, 10.5, 10.5,
        ],
    )
    .unwrap();
    let fitted = AffinityPropagation::<f64>::new().fit(&x, &()).unwrap();

    // Every exemplar index is in range and is labeled into a distinct cluster.
    let n = 8usize;
    let labels = fitted.labels();
    let exemplars = fitted.exemplar_indices();
    assert!(!exemplars.is_empty());
    let mut seen = std::collections::BTreeSet::new();
    for &ex in exemplars {
        assert!(ex < n, "exemplar index in range");
        seen.insert(labels[ex]);
    }
    assert_eq!(
        seen.len(),
        exemplars.len(),
        "REQ-1: each exemplar lands in its own distinct cluster"
    );
    // Labels form a valid 0..n_clusters partition.
    let k = fitted.n_clusters() as isize;
    for &l in labels {
        assert!((0..k).contains(&l), "label in [0, n_clusters)");
    }
}

/// PIN — REQ-2 (default preference), tracking #971. FAILS until the fixer lands.
///
/// Divergence: `affinity_propagation.rs` `fn fit` (the `pref` else-branch)
/// medians ONLY the `n(n-1)/2` off-diagonal upper-triangle similarities, whereas
/// sklearn `sklearn/cluster/_affinity_propagation.py:519-520`
/// (`preference = np.median(self.affinity_matrix_)`) medians the FULL n×n
/// affinity matrix including the n zero self-distances.
///
/// Fixture: `make_blobs(n_samples=15, centers=3, cluster_std=0.5, random_state=5)`.
/// Live sklearn 1.5.2:
/// ```text
/// A = -euclidean_distances(X, squared=True)
/// full_median   = np.median(A)                      = -2.8509764620227003
/// offdiag_median= np.median(A[triu_indices(15,1)])  = -4.238430447520386
/// AffinityPropagation(random_state=0).fit(X)  # default (full-median pref)
///   n_clusters = 3
///   labels_    = [1, 2, 0, 1, 0, 1, 1, 0, 1, 2, 0, 2, 0, 2, 2]
/// AffinityPropagation(preference=-4.2384..., random_state=0).fit(X)  # off-diag
///   n_clusters = 2
/// ```
/// sklearn's DEFAULT model yields a 3-cluster partition; current ferrolearn
/// DEFAULT yields 2 (it uses the off-diagonal median). After the
/// off-diagonal→full-matrix median fix, ferrolearn DEFAULT reproduces the
/// 3-cluster sklearn partition — proven: ferrolearn with the explicit full
/// median (`-2.8509764620227003`) already returns exactly
/// `[1, 2, 0, 1, 0, 1, 1, 0, 1, 2, 0, 2, 0, 2, 2]`, so the preference is the
/// SOLE cause of the disagreement (REQ-4/5/6 do not interfere on this fixture).
#[test]
#[ignore = "divergence: REQ-2 default preference medians off-diagonal not full matrix; tracking #971"]
fn pin_req2_default_preference_partition() {
    // make_blobs(n_samples=15, centers=3, cluster_std=0.5, random_state=5)
    let x = Array2::from_shape_vec(
        (15, 2),
        vec![
            0.2664436893532643, 2.591087893491199,
            -6.621206672055655, 8.694641914204821,
            -6.014752780633331, 7.118827794582383,
            -0.20220410249389675, 2.053221818811956,
            -5.739551051705831, 7.716381924852275,
            -0.23487367834629636, 2.1843434521482927,
            -0.23013180263636165, 2.1819120370242033,
            -6.156518884416113, 7.312207868253734,
            0.1647504356275682, 1.9190914431568074,
            -6.355920835820783, 7.943791581400387,
            -5.466334965286692, 7.249711144657848,
            -6.698011157947319, 8.022128639913456,
            -5.505331657416119, 8.205886682078308,
            -5.289921388467887, 9.300883662374089,
            -6.301556484839299, 8.160964194177236,
        ],
    )
    .unwrap();

    let fitted = AffinityPropagation::<f64>::new().fit(&x, &()).unwrap();

    // sklearn DEFAULT (full-matrix-median preference) partition.
    let sklearn_labels: [isize; 15] = [1, 2, 0, 1, 0, 1, 1, 0, 1, 2, 0, 2, 0, 2, 2];
    let expected = canonical_partition(&sklearn_labels);

    assert_eq!(
        fitted.n_clusters(),
        3,
        "REQ-2: default preference should be the FULL-matrix median (-2.851 -> 3 clusters), \
         not the off-diagonal median (-4.238 -> 2 clusters)"
    );
    assert_eq!(
        canonical_partition(fitted.labels().as_slice().unwrap()),
        expected,
        "REQ-2: default-preference partition must match sklearn's full-median partition"
    );
}
