//! Divergence + green-guard suite for `AgglomerativeClustering`
//! (`ferrolearn-cluster/src/agglomerative.rs`) vs scikit-learn 1.5.2
//! `sklearn/cluster/_agglomerative.py`
//! (`class AgglomerativeClustering(ClusterMixin, BaseEstimator)`, `:781`).
//!
//! Design doc: `.design/cluster/agglomerative.md` (12 REQs; 4 SHIPPED).
//!
//! ## Green guards (PASS against current code)
//!
//! These pin the genuinely-SHIPPED PARTITION + `n_clusters_` contracts
//! (REQ-1/REQ-2/REQ-3). They are PERMUTATION-INVARIANT: the absolute
//! `labels_` integer numbering DIVERGES from sklearn's `_hc_cut` heap
//! enumeration (REQ-7, blocker #938) and is intentionally NOT asserted.
//! Each expected partition is the canonicalized set of co-membership groups
//! produced by the live sklearn 1.5.2 oracle (R-CHAR-3 — never copied from
//! ferrolearn):
//!
//! ```text
//! python3 -c "import numpy as np; from sklearn.cluster import AgglomerativeClustering; \
//!   X2=np.array([[0.,0.],[0.1,0.],[0.,0.1],[0.05,0.05],[10.,10.],[10.1,10.],[10.,10.1],[10.05,10.05]]); \
//!   X3=np.array([[0.,0.],[0.1,0.1],[-0.1,0.1],[10.,10.],[10.1,10.1],[9.9,10.1],[0.,10.],[0.1,10.1],[-0.1,9.9]]); \
//!   [print(lk, AgglomerativeClustering(n_clusters=2,linkage=lk).fit(X2).labels_.tolist(), \
//!                AgglomerativeClustering(n_clusters=3,linkage=lk).fit(X3).labels_.tolist()) \
//!    for lk in ['ward','complete','average','single']]"
//! # all four linkages:
//! #   2-blob partition  -> {0,1,2,3} | {4,5,6,7}      (n_clusters_=2)
//! #   3-blob partition  -> {0,1,2} | {3,4,5} | {6,7,8} (n_clusters_=3)
//! ```
//!
//! ## No divergence pinned this iteration
//!
//! REQ-5 (`ensure_min_samples=2`, `_agglomerative.py:989`) is the prime
//! minimal-fix candidate, but it is NOT safely fixable in `agglomerative.rs`
//! alone: the in-crate consumer `birch.rs` (`fn fit`, global Ward clustering on
//! `subcluster_centers_`) legitimately calls
//! `AgglomerativeClustering::new(1).fit(<single-row>)` on its single-subcluster
//! path. Two birch tests (`test_identical_points`, `test_single_cluster`)
//! exercise exactly that path and assert success. (Notably, sklearn's OWN Birch
//! raises `ValueError` on the same inputs, so the birch reliance is itself a
//! divergence — but unwinding it is a coupled multi-file change, not a minimal
//! single-file fix.) Per the iteration's safety gate, REQ-5 stays NOT-STARTED
//! (blocker #964) and NOTHING is pinned RED here. See the critic report.

use ferrolearn_cluster::{AgglomerativeClustering, Linkage};
use ferrolearn_core::Fit;
use ndarray::Array2;

/// Two well-separated blobs — identical to `agglomerative.rs::make_two_blobs`.
fn two_blobs() -> Array2<f64> {
    Array2::from_shape_vec(
        (8, 2),
        vec![
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.05, 0.05, 10.0, 10.0, 10.1, 10.0, 10.0, 10.1, 10.05,
            10.05,
        ],
    )
    .unwrap()
}

/// Three well-separated blobs — identical to `agglomerative.rs::make_three_blobs`.
fn three_blobs() -> Array2<f64> {
    Array2::from_shape_vec(
        (9, 2),
        vec![
            0.0, 0.0, 0.1, 0.1, -0.1, 0.1, 10.0, 10.0, 10.1, 10.1, 9.9, 10.1, 0.0, 10.0, 0.1, 10.1,
            -0.1, 9.9,
        ],
    )
    .unwrap()
}

/// Canonicalize a label vector into a permutation-invariant partition:
/// the sorted list of (sorted sample-index group) tuples. This collapses
/// any label-permutation so the PARTITION can be compared regardless of the
/// absolute integers (REQ-1 PARTITION contract vs REQ-7 absolute numbering).
fn canonical_partition(labels: &[usize]) -> Vec<Vec<usize>> {
    use std::collections::BTreeMap;
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, &l) in labels.iter().enumerate() {
        groups.entry(l).or_default().push(i);
    }
    let mut parts: Vec<Vec<usize>> = groups.into_values().collect();
    for g in &mut parts {
        g.sort_unstable();
    }
    parts.sort();
    parts
}

fn fit_partition(x: &Array2<f64>, n_clusters: usize, linkage: Linkage) -> (Vec<Vec<usize>>, usize) {
    let fitted = AgglomerativeClustering::<f64>::new(n_clusters)
        .with_linkage(linkage)
        .fit(x, &())
        .expect("fit must succeed on separable blobs");
    let labels: Vec<usize> = fitted.labels().iter().copied().collect();
    (canonical_partition(&labels), fitted.n_clusters())
}

// ─────────────────────────────────────────────────────────────────────────────
// Green guard — REQ-1/REQ-2/REQ-3: 2-blob partition + n_clusters_, all linkages
// sklearn oracle: partition {0,1,2,3}|{4,5,6,7}, n_clusters_=2 (all 4 linkages).
// ─────────────────────────────────────────────────────────────────────────────

/// Green guard (REQ-1/REQ-3, partition up-to-permutation): on the 2-blob
/// fixture every linkage yields the sklearn 1.5.2 partition
/// `{0,1,2,3} | {4,5,6,7}` (oracle command in module doc). REQ-2: `n_clusters_`
/// == requested 2 (`_agglomerative.py:1095`). PASSES against current code.
#[test]
fn green_two_blobs_partition_all_linkages() {
    // Expected = canonicalized sklearn 1.5.2 partition (live oracle), NOT
    // copied from ferrolearn.
    let expected: Vec<Vec<usize>> = vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]];
    let x = two_blobs();
    for linkage in [
        Linkage::Ward,
        Linkage::Complete,
        Linkage::Average,
        Linkage::Single,
    ] {
        let (partition, n_clusters_) = fit_partition(&x, 2, linkage);
        assert_eq!(
            partition, expected,
            "2-blob partition mismatch for linkage {linkage:?} (sklearn = {expected:?})"
        );
        assert_eq!(
            n_clusters_, 2,
            "n_clusters_ must equal requested 2 for linkage {linkage:?}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Green guard — REQ-1/REQ-2/REQ-3: 3-blob partition + n_clusters_, all linkages
// sklearn oracle: partition {0,1,2}|{3,4,5}|{6,7,8}, n_clusters_=3 (all 4).
// ─────────────────────────────────────────────────────────────────────────────

/// Green guard (REQ-1/REQ-3, partition up-to-permutation): on the 3-blob
/// fixture every linkage yields the sklearn 1.5.2 partition
/// `{0,1,2} | {3,4,5} | {6,7,8}` (oracle command in module doc). REQ-2:
/// `n_clusters_` == requested 3. PASSES against current code. The ABSOLUTE
/// numbering (sklearn ward `[2,2,2,1,1,1,0,0,0]`) is REQ-7/#938 and is
/// deliberately NOT asserted — only co-membership is.
#[test]
fn green_three_blobs_partition_all_linkages() {
    // Expected = canonicalized sklearn 1.5.2 partition (live oracle).
    let expected: Vec<Vec<usize>> = vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8]];
    let x = three_blobs();
    for linkage in [
        Linkage::Ward,
        Linkage::Complete,
        Linkage::Average,
        Linkage::Single,
    ] {
        let (partition, n_clusters_) = fit_partition(&x, 3, linkage);
        assert_eq!(
            partition, expected,
            "3-blob partition mismatch for linkage {linkage:?} (sklearn = {expected:?})"
        );
        assert_eq!(
            n_clusters_, 3,
            "n_clusters_ must equal requested 3 for linkage {linkage:?}"
        );
    }
}
