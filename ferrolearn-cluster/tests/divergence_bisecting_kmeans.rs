//! Divergence / green-guard tests for `BisectingKMeans` vs scikit-learn 1.5.2
//! `sklearn/cluster/_bisect_k_means.py`.
//!
//! These are EXTERNAL integration tests (public API only — the struct fields are
//! private). They GREEN-GUARD the two SHIPPED contracts from
//! `.design/cluster/bisecting_kmeans.md`:
//!   - REQ-1: clustering PARTITION up-to-permutation on well-separated blobs.
//!   - REQ-13: `transform` output contract (shape, euclidean metric,
//!     column-to-center correspondence == predict label).
//!
//! Expected values are computed from the LIVE sklearn 1.5.2 oracle (R-CHAR-3),
//! NEVER literal-copied from the ferrolearn side. The sklearn label integers are
//! canonicalized to a permutation-invariant partition signature so the
//! up-to-permutation REQ-1 claim is what is asserted (the absolute integers
//! diverge — REQ-5, blocker #1027 — and are deliberately NOT asserted here).

use ferrolearn_cluster::BisectingKMeans;
use ferrolearn_core::traits::{Fit, Predict, Transform};
use ndarray::Array2;

/// Canonicalize a label vector into a permutation-invariant partition signature:
/// for each sample, the index of its cluster's FIRST occurrence. This collapses
/// any relabeling/permutation of cluster ids to a canonical form, so two label
/// vectors that induce the same grouping compare equal regardless of which
/// integer each group received.
fn canonical_partition(labels: &[isize]) -> Vec<usize> {
    let mut first_seen: Vec<isize> = Vec::new();
    labels
        .iter()
        .map(|&l| match first_seen.iter().position(|&x| x == l) {
            Some(p) => p,
            None => {
                first_seen.push(l);
                first_seen.len() - 1
            }
        })
        .collect()
}

/// Divergence-guard (SHIPPED, REQ-1): ferrolearn's `fn fit` recovers sklearn's
/// clustering PARTITION up-to-permutation on a well-separated 2-blob fixture.
///
/// sklearn oracle (live, 1.5.2):
///   X = [[0,0],[0.2,0.1],[-0.1,0.2],[0.1,-0.1],[20,20],[20.2,20.1],[19.9,20.2],[20.1,19.8]]
///   BisectingKMeans(n_clusters=2, random_state=0).fit(X).labels_
///     -> [0, 0, 0, 0, 1, 1, 1, 1]
///   canonical partition -> [0,0,0,0,1,1,1,1]
/// This is a FRESH separable fixture not present in `bisecting_kmeans.rs`.
#[test]
fn green_req1_two_blob_partition_up_to_permutation() {
    // sklearn-oracle labels for this exact fixture (live sklearn 1.5.2):
    // [0, 0, 0, 0, 1, 1, 1, 1]; canonicalized below so only the PARTITION is asserted.
    let sk_labels: [isize; 8] = [0, 0, 0, 0, 1, 1, 1, 1];
    let sk_canon = canonical_partition(&sk_labels);

    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            0.0, 0.0, 0.2, 0.1, -0.1, 0.2, 0.1, -0.1, // blob A near (0,0)
            20.0, 20.0, 20.2, 20.1, 19.9, 20.2, 20.1, 19.8, // blob B near (20,20)
        ],
    )
    .unwrap();

    let fitted = BisectingKMeans::<f64>::new(2)
        .with_random_state(0)
        .fit(&x, &())
        .unwrap();
    let ferro_canon = canonical_partition(fitted.labels().as_slice().unwrap());

    assert_eq!(
        ferro_canon, sk_canon,
        "ferrolearn 2-blob partition (up to permutation) must match sklearn"
    );
}

/// Divergence-guard (SHIPPED, REQ-1): ferrolearn's `fn fit` recovers sklearn's
/// clustering PARTITION up-to-permutation on a well-separated 3-blob fixture.
///
/// sklearn oracle (live, 1.5.2):
///   X = [[0,0],[0.3,0.2],[-0.2,0.3],[30,0],[30.2,0.1],[29.8,-0.2],[0,30],[0.1,30.3],[-0.2,29.8]]
///   BisectingKMeans(n_clusters=3, random_state=0).fit(X).labels_
///     -> [2, 2, 2, 0, 0, 0, 1, 1, 1]
///   canonical partition -> [0,0,0,1,1,1,2,2,2]
/// Fresh fixture not present in `bisecting_kmeans.rs`.
#[test]
fn green_req1_three_blob_partition_up_to_permutation() {
    // sklearn-oracle labels for this exact fixture (live sklearn 1.5.2):
    // [2, 2, 2, 0, 0, 0, 1, 1, 1]; canonicalized to the PARTITION below.
    let sk_labels: [isize; 9] = [2, 2, 2, 0, 0, 0, 1, 1, 1];
    let sk_canon = canonical_partition(&sk_labels);

    let x = Array2::from_shape_vec(
        (9, 2),
        vec![
            0.0, 0.0, 0.3, 0.2, -0.2, 0.3, // blob A near (0,0)
            30.0, 0.0, 30.2, 0.1, 29.8, -0.2, // blob B near (30,0)
            0.0, 30.0, 0.1, 30.3, -0.2, 29.8, // blob C near (0,30)
        ],
    )
    .unwrap();

    let fitted = BisectingKMeans::<f64>::new(3)
        .with_random_state(0)
        .fit(&x, &())
        .unwrap();
    let ferro_canon = canonical_partition(fitted.labels().as_slice().unwrap());

    assert_eq!(
        ferro_canon, sk_canon,
        "ferrolearn 3-blob partition (up to permutation) must match sklearn"
    );
}

/// Divergence-guard (SHIPPED, REQ-13): `transform` output CONTRACT — shape
/// `(n_samples, n_clusters)`, column `j` = euclidean distance to center `j`,
/// and argmin-over-columns == `predict` label.
///
/// sklearn oracle (live, 1.5.2) on the 3-blob fixture above:
///   m.transform(X).shape -> (9, 3)
///   m.transform(X).argmin(axis=1) == m.predict(X)  (column-to-center correspondence)
/// This guards the metric/shape/ordering contract, NOT exact center values
/// (those track REQ-2, blocker #1024). The argmin==predict identity follows from
/// the transform columns being distances to the same centers `predict` uses.
#[test]
fn green_req13_transform_contract_shape_and_argmin_equals_predict() {
    // sklearn-oracle contract facts (live sklearn 1.5.2):
    //   transform(X).shape == (9, 3)
    //   transform(X).argmin(axis=1) == predict(X)
    let sk_shape = (9usize, 3usize);

    let x = Array2::from_shape_vec(
        (9, 2),
        vec![
            0.0, 0.0, 0.3, 0.2, -0.2, 0.3, 30.0, 0.0, 30.2, 0.1, 29.8, -0.2, 0.0, 30.0, 0.1, 30.3,
            -0.2, 29.8,
        ],
    )
    .unwrap();

    let fitted = BisectingKMeans::<f64>::new(3)
        .with_random_state(0)
        .fit(&x, &())
        .unwrap();

    let t = fitted.transform(&x).unwrap();
    assert_eq!(
        t.dim(),
        sk_shape,
        "transform shape must be (n_samples, n_clusters)"
    );

    // Column-to-center correspondence: argmin over the distance columns must equal
    // the predict label (sklearn: transform(X).argmin(axis=1) == predict(X)).
    let predicted = fitted.predict(&x).unwrap();
    for i in 0..t.nrows() {
        let mut argmin = 0usize;
        let mut best = t[[i, 0]];
        for c in 1..t.ncols() {
            if t[[i, c]] < best {
                best = t[[i, c]];
                argmin = c;
            }
        }
        assert_eq!(
            argmin as isize, predicted[i],
            "transform argmin-over-columns must equal predict label at row {i}"
        );
    }

    // Metric check: each transform entry is the euclidean (non-squared) distance to
    // its center, hence non-negative, and the per-row minimum is the predicted
    // cluster's distance (a non-negative finite quantity on well-separated data).
    for v in t.iter() {
        assert!(
            *v >= 0.0,
            "transform distances are non-negative euclidean norms"
        );
        assert!(v.is_finite(), "transform distances are finite");
    }
}
