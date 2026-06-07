//! VALUE-parity tests for `FeatureAgglomeration` / `FittedFeatureAgglomeration`
//! (`ferrolearn-cluster/src/feature_agglomeration.rs`) against the LIVE
//! scikit-learn 1.5.2 oracle (`from sklearn.cluster import FeatureAgglomeration`,
//! mirroring `sklearn/cluster/_feature_agglomeration.py` +
//! `sklearn/cluster/_agglomerative.py`).
//!
//! These pin the END-TO-END VALUE win that landed once the AgglomerativeClustering
//! `_hc_cut` `labels_` numbering shipped (#938, commit 3e001cf4b):
//! `FeatureAgglomeration::fit` delegates to `AgglomerativeClustering::fit(X.T)`
//! (`_agglomerative.py:1339`), so the now bit-exact `labels_` flow straight through,
//! and `fn transform` (which groups by ASCENDING label index) emits VALUE-EXACT,
//! COLUMN-ORDERED output matching sklearn.
//!
//! Every expected value below is a LIVE `sklearn` 1.5.2 oracle value (computed via
//! `python3 -c "..."` run from `/tmp`, quoted above each block) — NEVER copied from
//! the ferrolearn side (goal.md R-CHAR-3). The companion file
//! `divergence_feature_agglomeration.rs` holds the validation pins + as-set guards;
//! this file asserts FULL VALUE + COLUMN-ORDER + delegated-attr equality.
//!
//! Live oracle setup (sklearn 1.5.2, run from /tmp):
//!   import numpy as np; from sklearn.cluster import FeatureAgglomeration
//!   X=np.array([[1.,1.1,5.,5.1,9.,9.1],[2.,2.1,6.,6.1,8.,8.1],
//!               [3.,3.1,7.,7.1,7.,7.1],[4.,4.1,8.,8.1,6.,6.1],
//!               [5.,5.1,9.,9.1,5.,5.1]])
//! Design doc: `.design/cluster/feature_agglomeration.md`.

use ferrolearn_cluster::{AgglomerativeLinkage, FeatureAgglomeration, PoolingFunc};
use ferrolearn_core::traits::{Fit, Transform};
use ndarray::Array2;

/// The 5x6 fixture: 6 features paired `(0,1)`/`(2,3)`/`(4,5)` so the `_hc_cut`
/// numbering is non-trivial (three feature groups, k∈{2,3} are distinct).
/// Identical matrix to the design-doc live-oracle probes.
fn make_correlated_features() -> Array2<f64> {
    Array2::from_shape_vec(
        (5, 6),
        vec![
            1.0, 1.1, 5.0, 5.1, 9.0, 9.1, 2.0, 2.1, 6.0, 6.1, 8.0, 8.1, 3.0, 3.1, 7.0, 7.1, 7.0,
            7.1, 4.0, 4.1, 8.0, 8.1, 6.0, 6.1, 5.0, 5.1, 9.0, 9.1, 5.0, 5.1,
        ],
    )
    .unwrap()
}

// ===========================================================================
// REQ-3 / REQ-5 — labels_ integer-EXACT for all four linkages, k in {2,3}.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   for L in ['ward','complete','average','single']:
//     for k in [2,3]:
//       m=FeatureAgglomeration(n_clusters=k, linkage=L).fit(X)
//       print(L, k, m.labels_.tolist())
//   -> EVERY linkage:  k=2 -> [1, 1, 0, 0, 0, 0]
//                      k=3 -> [0, 0, 2, 2, 1, 1]
//
// (The integer index — not just the partition — is bit-exact post-#938: the
// inner AgglomerativeClustering `_hc_cut` numbering, `_agglomerative.py:1099`
// + `np.searchsorted` `:1105`, flows through `fit`'s X.T delegation.)
// ===========================================================================

/// `FeatureAgglomeration(n_clusters=k, linkage=L).fit(X).labels_` is integer-EXACT
/// vs sklearn for all four linkages {ward,complete,average,single} and k∈{2,3}.
/// sklearn oracle (above): k=2 → `[1,1,0,0,0,0]`, k=3 → `[0,0,2,2,1,1]` (same
/// across all four linkages on this fixture). Asserts both `labels()` (sklearn
/// name) and `feature_labels()` (alias) return this exact numbering.
#[test]
fn value_labels_exact_all_linkages_k2_k3() {
    let x = make_correlated_features();

    // sklearn oracle, per (linkage, k). All four linkages agree on this fixture.
    let expected_k2: [usize; 6] = [1, 1, 0, 0, 0, 0];
    let expected_k3: [usize; 6] = [0, 0, 2, 2, 1, 1];

    for linkage in [
        AgglomerativeLinkage::Ward,
        AgglomerativeLinkage::Complete,
        AgglomerativeLinkage::Average,
        AgglomerativeLinkage::Single,
    ] {
        for (k, expected) in [(2usize, &expected_k2[..]), (3usize, &expected_k3[..])] {
            let fitted = FeatureAgglomeration::<f64>::new(k)
                .with_linkage(linkage)
                .fit(&x, &())
                .unwrap();

            let labels: Vec<usize> = fitted.labels().to_vec();
            assert_eq!(
                labels, expected,
                "labels() for linkage={linkage:?} k={k} should equal sklearn \
                 FeatureAgglomeration(...).fit(X).labels_ {expected:?}, got {labels:?}"
            );

            // `feature_labels()` is the backward-compat alias — same data.
            let aliased: Vec<usize> = fitted.feature_labels().to_vec();
            assert_eq!(
                aliased, expected,
                "feature_labels() alias for linkage={linkage:?} k={k} should equal \
                 labels() / sklearn labels_ {expected:?}, got {aliased:?}"
            );
        }
    }
}

// ===========================================================================
// REQ-1 — transform mean-pooling: FULL VALUE + COLUMN ORDER vs sklearn.
//
// sklearn mean fast path (`_feature_agglomeration.py:51-57`):
//   size = np.bincount(labels_); nX[i] = np.bincount(labels_, X[i,:]) / size
//   -> column j = MEAN over features with labels_ == j, j ascending.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   m=FeatureAgglomeration(n_clusters=3, linkage='ward', pooling_func=np.mean).fit(X)
//   print(np.round(m.transform(X),12).tolist())
//   -> [[1.05, 9.05, 5.05],[2.05, 8.05, 6.05],[3.05, 7.05, 7.05],
//       [4.05, 6.05, 8.05],[5.05, 5.05, 9.05]]
//   (single linkage gives the SAME matrix on this fixture.)
// ===========================================================================

/// `FeatureAgglomeration(n_clusters=3, pooling_func=np.mean).fit(X).transform(X)`
/// is VALUE-EXACT and COLUMN-ORDERED vs sklearn — full-matrix equality including
/// column order (the REQ-1 win over the old as-set guard). sklearn oracle (above):
/// row0 `[1.05, 9.05, 5.05]` (NOTE the middle column is `9.05`, not `5.05` — column
/// order is by ascending label index, `labels_=[0,0,2,2,1,1]`). Verified for ward
/// and single linkage (identical matrix on this fixture).
#[test]
fn value_transform_mean_column_ordered() {
    // sklearn oracle: transform matrix, column-ordered by ascending label index.
    let expected: [[f64; 3]; 5] = [
        [1.05, 9.05, 5.05],
        [2.05, 8.05, 6.05],
        [3.05, 7.05, 7.05],
        [4.05, 6.05, 8.05],
        [5.05, 5.05, 9.05],
    ];

    let x = make_correlated_features();
    for linkage in [AgglomerativeLinkage::Ward, AgglomerativeLinkage::Single] {
        let fitted = FeatureAgglomeration::<f64>::new(3)
            .with_linkage(linkage)
            .with_pooling_func(PoolingFunc::Mean)
            .fit(&x, &())
            .unwrap();
        let reduced = fitted.transform(&x).unwrap();

        assert_eq!(reduced.dim(), (5, 3));
        for (i, exp_row) in expected.iter().enumerate() {
            for (j, &exp) in exp_row.iter().enumerate() {
                assert!(
                    (reduced[[i, j]] - exp).abs() < 1e-9,
                    "mean transform[{i},{j}] linkage={linkage:?}: ferrolearn {} vs \
                     sklearn {exp} (column-ordered VALUE)",
                    reduced[[i, j]]
                );
            }
        }
    }
}

// ===========================================================================
// REQ-2 — transform max-pooling: FULL VALUE + COLUMN ORDER vs sklearn.
//
// sklearn general path (`_feature_agglomeration.py:58-63`):
//   nX = [pooling_func(X[:, labels_ == l], axis=1) for l in np.unique(labels_)]
//   nX = np.array(nX).T  -> columns ordered by sorted unique label = 0..k-1.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   m=FeatureAgglomeration(n_clusters=3, linkage='ward', pooling_func=np.max).fit(X)
//   print(np.round(m.transform(X),12).tolist())
//   -> [[1.1, 9.1, 5.1],[2.1, 8.1, 6.1],[3.1, 7.1, 7.1],
//       [4.1, 6.1, 8.1],[5.1, 5.1, 9.1]]
//   (complete linkage gives the SAME matrix on this fixture.)
// ===========================================================================

/// `FeatureAgglomeration(n_clusters=3, pooling_func=np.max).fit(X).transform(X)` is
/// VALUE-EXACT and COLUMN-ORDERED vs sklearn — full-matrix equality including column
/// order (the REQ-2 win). sklearn oracle (above): row0 `[1.1, 9.1, 5.1]`. Verified
/// for ward and complete linkage (identical matrix on this fixture).
#[test]
fn value_transform_max_column_ordered() {
    // sklearn oracle: transform matrix (max pooling), column-ordered.
    let expected: [[f64; 3]; 5] = [
        [1.1, 9.1, 5.1],
        [2.1, 8.1, 6.1],
        [3.1, 7.1, 7.1],
        [4.1, 6.1, 8.1],
        [5.1, 5.1, 9.1],
    ];

    let x = make_correlated_features();
    for linkage in [AgglomerativeLinkage::Ward, AgglomerativeLinkage::Complete] {
        let fitted = FeatureAgglomeration::<f64>::new(3)
            .with_linkage(linkage)
            .with_pooling_func(PoolingFunc::Max)
            .fit(&x, &())
            .unwrap();
        let reduced = fitted.transform(&x).unwrap();

        assert_eq!(reduced.dim(), (5, 3));
        for (i, exp_row) in expected.iter().enumerate() {
            for (j, &exp) in exp_row.iter().enumerate() {
                assert!(
                    (reduced[[i, j]] - exp).abs() < 1e-9,
                    "max transform[{i},{j}] linkage={linkage:?}: ferrolearn {} vs \
                     sklearn {exp} (column-ordered VALUE)",
                    reduced[[i, j]]
                );
            }
        }
    }
}

// ===========================================================================
// REQ-9 — delegated fitted attrs: children_, distances_, n_leaves_,
//         n_connected_components_ over X.T.
//
// sklearn `FeatureAgglomeration._fit` -> `AgglomerativeClustering._fit(X.T)`
// (`_agglomerative.py:1339`) sets these from the inner clustering.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   m=FeatureAgglomeration(n_clusters=3, linkage='ward',
//                          compute_distances=True).fit(X)
//   print(m.children_.tolist(), m.distances_.tolist(),
//         m.n_leaves_, m.n_connected_components_)
//   -> children_ = [[2,3],[4,5],[0,1],[6,7],[8,9]]
//      distances_ = [0.22360679775, 0.22360679775, 0.22360679775,
//                    8.944271909999, 15.49193338483]
//      n_leaves_ = 6,  n_connected_components_ = 1
// ===========================================================================

/// Delegated fitted attrs match sklearn `FeatureAgglomeration(..., linkage='ward',
/// compute_distances=True).fit(X)`: `children_` == `[[2,3],[4,5],[0,1],[6,7],[8,9]]`
/// (as `(usize,usize)` pairs), `distances_` ~1e-9 == sklearn `distances_`,
/// `n_leaves_ == 6 == n_features`, `n_connected_components_ == 1`. `distances()` is
/// `None` without `with_compute_distances(true)` (mirrors sklearn `compute_distances`
/// default `False`).
#[test]
fn value_fitted_attrs_delegated() {
    let x = make_correlated_features();

    // sklearn oracle (ward, compute_distances=True).
    let expected_children: [(usize, usize); 5] = [(2, 3), (4, 5), (0, 1), (6, 7), (8, 9)];
    let expected_distances: [f64; 5] = [
        0.223_606_797_75,
        0.223_606_797_75,
        0.223_606_797_75,
        8.944_271_909_999,
        15.491_933_384_83,
    ];

    let fitted = FeatureAgglomeration::<f64>::new(3)
        .with_linkage(AgglomerativeLinkage::Ward)
        .with_compute_distances(true)
        .fit(&x, &())
        .unwrap();

    // children_ — bit-exact pair list.
    assert_eq!(
        fitted.children(),
        &expected_children[..],
        "children_ should match sklearn FeatureAgglomeration(compute_distances=True) \
         children_ {expected_children:?}"
    );

    // distances_ — Some, ~1e-9 vs sklearn.
    let distances = fitted
        .distances()
        .expect("distances() should be Some when with_compute_distances(true)");
    assert_eq!(distances.len(), 5);
    for (i, &exp) in expected_distances.iter().enumerate() {
        assert!(
            (distances[i] - exp).abs() < 1e-9,
            "distances_[{i}]: ferrolearn {} vs sklearn {exp}",
            distances[i]
        );
    }

    // n_leaves_ == n_features == 6;  n_connected_components_ == 1.
    assert_eq!(
        fitted.n_leaves(),
        6,
        "n_leaves_ should equal n_features (6), sklearn oracle: 6"
    );
    assert_eq!(fitted.n_leaves(), fitted.n_features());
    assert_eq!(
        fitted.n_connected_components(),
        1,
        "n_connected_components_ should be 1 (unstructured path), sklearn oracle: 1"
    );

    // Without compute_distances, distances() is None (sklearn default).
    let no_dist = FeatureAgglomeration::<f64>::new(3)
        .with_linkage(AgglomerativeLinkage::Ward)
        .fit(&x, &())
        .unwrap();
    assert!(
        no_dist.distances().is_none(),
        "distances() should be None without with_compute_distances(true) \
         (sklearn compute_distances default False)"
    );
    // children_/n_leaves_/n_connected_components_ are always present.
    assert_eq!(no_dist.children(), &expected_children[..]);
    assert_eq!(no_dist.n_leaves(), 6);
    assert_eq!(no_dist.n_connected_components(), 1);
}

// ===========================================================================
// REQ-8 — inverse_transform broadcasts each cluster's pooled value back to its
// member features (`_feature_agglomeration.py:66-92`: `X[..., inverse]`).
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   for L in ['ward','complete','average','single']:
//     m=FeatureAgglomeration(n_clusters=3, linkage=L).fit(X)
//     print(L, m.inverse_transform(m.transform(X)).flatten().tolist())
//   # all four linkages (identical partition): row-major (5x6)
//   # [1.05,1.05, 5.05,5.05, 9.05,9.05,  2.05,2.05, 6.05,6.05, 8.05,8.05,
//   #  3.05,3.05, 7.05,7.05, 7.05,7.05,  4.05,4.05, 8.05,8.05, 6.05,6.05,
//   #  5.05,5.05, 9.05,9.05, 5.05,5.05]
// ===========================================================================

/// REQ-8: `inverse_transform(transform(X))` broadcasts the per-cluster mean
/// back to every feature in the cluster, matching sklearn bit-for-bit for all
/// four linkages. Also checks the shape `(n_samples, n_features)` and the
/// `xred.ncols() != n_clusters` shape-mismatch rejection.
#[test]
fn value_inverse_transform_roundtrip_and_broadcast() {
    // Live sklearn oracle: row-major (5, 6), identical across all four linkages.
    const SK_INV: [f64; 30] = [
        1.05, 1.05, 5.05, 5.05, 9.05, 9.05, 2.05, 2.05, 6.05, 6.05, 8.05, 8.05, 3.05, 3.05, 7.05,
        7.05, 7.05, 7.05, 4.05, 4.05, 8.05, 8.05, 6.05, 6.05, 5.05, 5.05, 9.05, 9.05, 5.05, 5.05,
    ];

    let x = make_correlated_features();

    for linkage in [
        AgglomerativeLinkage::Ward,
        AgglomerativeLinkage::Complete,
        AgglomerativeLinkage::Average,
        AgglomerativeLinkage::Single,
    ] {
        let fitted = FeatureAgglomeration::<f64>::new(3)
            .with_linkage(linkage)
            .with_pooling_func(PoolingFunc::Mean)
            .fit(&x, &())
            .unwrap();

        let xt = fitted.transform(&x).unwrap();
        let xinv = fitted.inverse_transform(&xt).unwrap();

        assert_eq!(
            xinv.dim(),
            (5, 6),
            "inverse_transform output must be (n_samples, n_features) = (5, 6), got {:?}",
            xinv.dim()
        );
        for (idx, &sk) in SK_INV.iter().enumerate() {
            let (i, f) = (idx / 6, idx % 6);
            assert!(
                (xinv[[i, f]] - sk).abs() < 1e-12,
                "{linkage:?}: inverse_transform[{i},{f}] = {} != sklearn {sk}",
                xinv[[i, f]]
            );
        }
    }

    // Shape-mismatch: too FEW columns is rejected (sklearn raises IndexError).
    // A WIDER xred is accepted with trailing columns ignored — see the #2187
    // pin in divergence_feature_agglom_inverse_transform.rs.
    let fitted = FeatureAgglomeration::<f64>::new(3)
        .with_linkage(AgglomerativeLinkage::Ward)
        .fit(&x, &())
        .unwrap();
    let bad = Array2::<f64>::zeros((5, 2)); // 2 < n_clusters (3): too few columns
    assert!(
        fitted.inverse_transform(&bad).is_err(),
        "inverse_transform must reject xred.ncols() < n_clusters (sklearn IndexError)"
    );
}
