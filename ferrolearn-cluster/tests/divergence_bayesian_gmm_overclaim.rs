//! RED divergence pin: the REQ-1 green guard
//! `green_req1_two_blob_comembership_matches_sklearn` in
//! `divergence_bayesian_gmm.rs` asserts a FABRICATED sklearn oracle value.
//!
//! That guard hardcodes the "sklearn" 2-blob partition as
//! `[[0,1,2,3],[4,5,6,7]]` (two distinct clusters). The LIVE sklearn 1.5.2
//! oracle on the SAME fixture does NOT split the two blobs — it prunes to a
//! single surviving component and labels all 8 points identically:
//!
//! ```text
//! python3 -c "import numpy as np; from sklearn.mixture import BayesianGaussianMixture; \
//!   y1=np.array([[0.,0.],[0.1,0.],[0.,0.1],[0.05,0.05]]); \
//!   y2=np.array([[20.,20.],[20.1,20.],[20.,20.1],[20.05,20.05]]); \
//!   Y=np.vstack([y1,y2]); \
//!   m=BayesianGaussianMixture(n_components=5,random_state=0).fit(Y); \
//!   print(m.fit_predict(Y).tolist())"
//! # [1, 1, 1, 1, 1, 1, 1, 1]   -> canonical [[0,1,2,3,4,5,6,7]]
//! ```
//!
//! So sklearn's canonical partition is `[[0,1,2,3,4,5,6,7]]` (one group),
//! NOT `[[0,1,2,3],[4,5,6,7]]`. ferrolearn's heuristic plain-EM DOES split the
//! two blobs (it does not prune), so ferrolearn DIVERGES from sklearn on this
//! exact fixture. The REQ-1 SHIPPED row's "recover sklearn's co-membership"
//! claim is therefore FALSE for the 2-blob case, and the green guard is an
//! R-CHAR-3 violation (expected value not from the live oracle; it matches the
//! ferrolearn side, not sklearn).

use ferrolearn_cluster::BayesianGaussianMixture;

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

fn two_blobs() -> ndarray::Array2<f64> {
    ndarray::Array2::from_shape_vec(
        (8, 2),
        vec![
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.05, 0.05, 20.0, 20.0, 20.1, 20.0, 20.0, 20.1, 20.05,
            20.05,
        ],
    )
    .unwrap()
}

/// Divergence: ferrolearn `BayesianGaussianMixture::fit_predict` diverges from
/// `sklearn/mixture/_base.py:286` (final e-step argmax) for the 2-blob fixture
/// used by `green_req1_two_blob_comembership_matches_sklearn`.
///
/// Live sklearn 1.5.2 oracle returns canonical partition `[[0,1,2,3,4,5,6,7]]`
/// (single component after pruning). ferrolearn returns `[[0,1,2,3],[4,5,6,7]]`
/// (two components, no pruning). This test asserts the REAL live-oracle value
/// and therefore FAILS against current ferrolearn — pinning the overclaim in
/// the REQ-1 SHIPPED row.
///
/// Tracking: #1067
#[test]
#[ignore = "divergence: REQ-1 green guard hardcodes a fabricated 2-blob sklearn partition (sklearn prunes to 1 cluster, ferrolearn splits to 2); tracking #1067"]
fn divergence_two_blob_partition_real_sklearn() {
    // From the LIVE sklearn 1.5.2 oracle (run from /tmp, random_state=0):
    //   fit_predict -> [1,1,1,1,1,1,1,1]  -> canonical [[0..7]]
    let sklearn_partition: Vec<Vec<usize>> = vec![vec![0, 1, 2, 3, 4, 5, 6, 7]];

    let x = two_blobs();
    let labels = BayesianGaussianMixture::<f64>::new(5)
        .with_random_state(0)
        .fit_predict(&x)
        .unwrap();
    let ferro_partition = canonical_partition(labels.as_slice().unwrap());

    assert_eq!(
        ferro_partition, sklearn_partition,
        "ferrolearn 2-blob partition {ferro_partition:?} must equal the LIVE \
         sklearn 1.5.2 partition {sklearn_partition:?} (single pruned component); \
         the existing green guard hardcodes a fabricated [[0,1,2,3],[4,5,6,7]]"
    );
}
