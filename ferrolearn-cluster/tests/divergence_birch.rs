//! Value-contract guards + (any) divergence pins for `Birch` /
//! `FittedBirch` (`ferrolearn-cluster/src/birch.rs`) against the LIVE
//! scikit-learn 1.5.2 oracle (`from sklearn.cluster import Birch`, mirroring
//! `sklearn/cluster/_birch.py`).
//!
//! Every expected value below is a LIVE `sklearn` 1.5.2 oracle value (computed
//! via `python3 -c "..."` run from `/tmp`, quoted above each block) — NEVER
//! copied from the ferrolearn side (goal.md R-CHAR-3).
//!
//! Design doc: `.design/cluster/birch.md` (commit 835f8240).
//!
//! ## Why these are GREEN (not RED pins)
//!
//! ferrolearn's `subcluster_centers_` (CF centroid `LS/N`,
//! `_birch.py:590-591`) and the `labels_` PARTITION (global Agglomerative-Ward
//! step, `_global_clustering`, `_birch.py:703-738`) genuinely VALUE-MATCH the
//! live oracle — but ONLY in the regime where the number of leaf subclusters
//! stays `<= branching_factor`. In that regime the flat-`Vec` CF structure
//! produces the SAME subclusters sklearn's CF-tree produces, so the centroid
//! VALUES (REQ-1) and the partition (REQ-3) coincide. These guards pin exactly
//! that in-regime parity:
//!
//! - `green_birch_subcluster_centers_blobs8` (REQ-1, `blobs8`) — the two
//!   centroid VALUES match exactly (compared as a SET; sklearn leaf-order may
//!   differ from ferrolearn insertion-order — REQ-1 is "centroid VALUES match",
//!   not row order).
//! - `green_birch_labels_partition_blobs8` (REQ-3, `blobs8`) — same PARTITION
//!   (co-membership), exactly 2 clusters. NOT absolute label equality: sklearn
//!   `[1,1,1,1,0,0,0,0]` vs ferrolearn label-swapped is a permutation.
//! - `green_birch_subcluster_centers_blobs20` (REQ-1, `blobs20`) — all 10
//!   centroid VALUES match as a SET to ~1e-9.
//! - `green_birch_threshold_validation` (REQ-10/REQ-4 bound) — `threshold=0`
//!   and `branching_factor=1` are rejected (sklearn `InvalidParameterError`).
//!
//! ## Documented NOT-STARTED surface (large structural / feature additions,
//! NOT minimal single-file fixes — NO forced failing test, per R-DEFER-6 + the
//! kdtree #831 / balltree #858 / DBSCAN #952 convention):
//!
//! REQ-2 — CF-tree leaf splitting (#954): ferrolearn `build_cf_tree` keeps a
//! FLAT `Vec<ClusteringFeature>` capped at `branching_factor` and MERGES the
//! two closest CFs on overflow (`find_closest_pair` + `merge`). sklearn
//! `_CFNode.insert_cf_subcluster` + `_split_node` (`_birch.py:48-263`) split a
//! full leaf into two leaves and GROW the tree, so leaf-subcluster count is
//! unbounded by `branching_factor`. Live oracle (sklearn 1.5.2, run from /tmp):
//!   python3 -c "import numpy as np; from sklearn.cluster import Birch
//!     X=np.random.RandomState(1).rand(60,2)*20
//!     b=Birch(n_clusters=None,threshold=1.0,branching_factor=5).fit(X)
//!     print(b.subcluster_centers_.shape[0])"
//!   ->  37   (ferrolearn caps at branching_factor = 5)
//! Root structural divergence — a CF-tree rewrite, not a minimal fix.
//! Documented only.
//!
//! REQ-4 — `n_clusters=3` default + estimator-instance form + error ABI (#955):
//! sklearn `Birch().n_clusters == 3` (`_birch.py:496`); ferrolearn
//! `Birch::new()` defaults `n_clusters = None`. sklearn accepts `None`/`int>=1`/
//! a `sklearn.cluster` estimator (`:486`, `:731-735`); ferrolearn `Option<usize>`
//! has no estimator form. Error TYPE diverges (sklearn `InvalidParameterError`
//! vs `FerroError::InvalidParameter`). Documented only.
//!
//! REQ-5 — `predict` / `transform` absent (#956): sklearn `_predict` =
//! `pairwise_distances_argmin(X, subcluster_centers_)` → `subcluster_labels_`
//! (`:651-679`); `transform` = `euclidean_distances(X, subcluster_centers_)`
//! (`:681-701`). `FittedBirch<F>` has neither. Documented only.
//!
//! REQ-6 — `subcluster_labels_` absent (#957): sklearn exposes the per-subcluster
//! global label (`:723`/`:735`); `FittedBirch` has no accessor. Documented only.
//!
//! REQ-7 — `not_enough_centroids` ConvergenceWarning vs silent clamp (#958):
//! sklearn, when `len(centroids) < n_clusters`, SKIPS the global step, sets
//! `subcluster_labels_ = arange`, warns `ConvergenceWarning` (`:716`/`:722-730`).
//! ferrolearn `fn fit` CLAMPS `k.min(n_subclusters)` and runs Agglomerative —
//! no warning, `n_clusters_` silently clamped. Documented only.
//!
//! REQ-8 — `partial_fit` absent (#959): sklearn `partial_fit` (`:613-638`)
//! incrementally inserts without rebuilding. ferrolearn offers only batch `fit`.
//! Documented only.
//!
//! REQ-9 — `compute_labels` / `copy` params absent (#960): sklearn
//! `compute_labels=True` (`:497`) + `copy=True` (`:498`). `Birch<F>` has neither.
//! Documented only.
//!
//! REQ-11 — ferray substrate (#961): `birch.rs` imports `ndarray` +
//! `num_traits::Float`, not `ferray-core`. Documented only.

use ferrolearn_cluster::Birch;
use ferrolearn_core::traits::Fit;
use ndarray::Array2;

/// Sort the rows of a centroid matrix into a canonical (lexicographic) order so
/// two matrices that match as a SET (same centroid VALUES, possibly different
/// row order) compare equal. REQ-1 is "centroid VALUES match", not row order —
/// sklearn's `subcluster_centers_` order follows its tree-leaf order; ferrolearn
/// follows insertion order.
fn sorted_rows(m: &Array2<f64>) -> Vec<Vec<f64>> {
    let mut rows: Vec<Vec<f64>> = m.outer_iter().map(|r| r.to_vec()).collect();
    rows.sort_by(|a, b| a.partial_cmp(b).unwrap());
    rows
}

// ===========================================================================
// GREEN — REQ-1, `blobs8`: subcluster_centers_ VALUES match as a SET.
//
// blobs8 = the make_two_blobs() fixture (8x2: four points near origin, four
// near (10,10)). With n_clusters=2, threshold=0.5, sklearn yields 2 leaf
// subclusters whose centroids are the per-blob means. 2 <= default
// branching_factor (50), so ferrolearn's flat Vec produces the SAME two CFs.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "import numpy as np; from sklearn.cluster import Birch
//     X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[0.05,0.05],
//                 [10.,10.],[10.1,10.],[10.,10.1],[10.05,10.05]])
//     b=Birch(n_clusters=2,threshold=0.5).fit(X)
//     print(b.subcluster_centers_.tolist())"
//   ->  [[0.037500000000000006, 0.037500000000000006],
//        [10.037500000000001, 10.037500000000001]]
// ===========================================================================

fn blobs8() -> Array2<f64> {
    Array2::from_shape_vec(
        (8, 2),
        vec![
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.05, 0.05, 10.0, 10.0, 10.1, 10.0, 10.0, 10.1, 10.05,
            10.05,
        ],
    )
    .unwrap()
}

/// Guard: ferrolearn `subcluster_centers()` on `blobs8` matches sklearn's
/// `subcluster_centers_` = `[[0.0375,0.0375],[10.0375,10.0375]]` (oracle above,
/// full f64) — compared as a SET (sorted rows), since centroid VALUES are the
/// REQ-1 contract, not row order.
#[test]
fn green_birch_subcluster_centers_blobs8() {
    // sklearn oracle subcluster_centers_ (full f64), as a set.
    let sklearn_centers: [[f64; 2]; 2] = [
        [0.037_500_000_000_000_006, 0.037_500_000_000_000_006],
        [10.037_500_000_000_001, 10.037_500_000_000_001],
    ];
    let mut sklearn_rows: Vec<Vec<f64>> = sklearn_centers.iter().map(|r| r.to_vec()).collect();
    sklearn_rows.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let x = blobs8();
    let fitted = Birch::<f64>::new()
        .with_threshold(0.5)
        .with_n_clusters(2)
        .fit(&x, &())
        .unwrap();
    let ferro_rows = sorted_rows(fitted.subcluster_centers());

    assert_eq!(
        ferro_rows.len(),
        sklearn_rows.len(),
        "blobs8 n_subclusters: ferrolearn {} vs sklearn {}",
        ferro_rows.len(),
        sklearn_rows.len()
    );
    for (i, (fr, sr)) in ferro_rows.iter().zip(sklearn_rows.iter()).enumerate() {
        for (j, (&f, &s)) in fr.iter().zip(sr.iter()).enumerate() {
            assert!(
                (f - s).abs() <= 1e-12,
                "blobs8 centroid[{i}][{j}] (sorted): ferrolearn {f} vs sklearn \
                 {s} (diff {})",
                (f - s).abs()
            );
        }
    }
}

// ===========================================================================
// GREEN — REQ-3, `blobs8`: labels_ PARTITION (co-membership) matches, 2 clusters.
//
// sklearn assigns the four origin points one label and the four (10,10) points
// another. Absolute label VALUES differ from ferrolearn by a permutation
// (Agglomerative-Ward label ordering), so we assert CO-MEMBERSHIP (points in
// the same sklearn cluster are in the same ferrolearn cluster, and vice versa)
// and exactly 2 distinct clusters — NOT element-wise label equality.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "import numpy as np; from sklearn.cluster import Birch
//     X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[0.05,0.05],
//                 [10.,10.],[10.1,10.],[10.,10.1],[10.05,10.05]])
//     print(Birch(n_clusters=2,threshold=0.5).fit_predict(X).tolist())"
//   ->  [1, 1, 1, 1, 0, 0, 0, 0]
// ===========================================================================

/// Guard: ferrolearn `fit_predict(&blobs8)` gives the SAME PARTITION as sklearn
/// `[1,1,1,1,0,0,0,0]` (oracle above): co-membership identical (i,j co-clustered
/// in sklearn <=> co-clustered in ferrolearn) and exactly 2 clusters. NOT
/// absolute label equality (sklearn vs ferrolearn differ by a label swap).
#[test]
fn green_birch_labels_partition_blobs8() {
    // sklearn oracle labels_ (Fixture blobs8).
    let sklearn_labels: [usize; 8] = [1, 1, 1, 1, 0, 0, 0, 0];

    let x = blobs8();
    let labels = Birch::<f64>::new()
        .with_threshold(0.5)
        .with_n_clusters(2)
        .fit_predict(&x)
        .unwrap();
    assert_eq!(labels.len(), 8);

    // Exactly 2 distinct clusters.
    let mut distinct: Vec<usize> = labels.iter().copied().collect();
    distinct.sort_unstable();
    distinct.dedup();
    assert_eq!(
        distinct.len(),
        2,
        "blobs8: ferrolearn produced {} distinct clusters, sklearn 2",
        distinct.len()
    );

    // Co-membership: for every pair, ferrolearn agrees with sklearn on whether
    // they share a cluster.
    for i in 0..8 {
        for j in (i + 1)..8 {
            let sk_same = sklearn_labels[i] == sklearn_labels[j];
            let fe_same = labels[i] == labels[j];
            assert_eq!(
                fe_same,
                sk_same,
                "blobs8 co-membership pair ({i},{j}): ferrolearn same={fe_same} \
                 vs sklearn same={sk_same} (labels ferro={:?} sklearn={:?})",
                labels.to_vec(),
                sklearn_labels
            );
        }
    }
}

// ===========================================================================
// GREEN — REQ-1, `blobs20`: all 10 subcluster_centers_ VALUES match as a SET.
//
// blobs20 = make_blobs(n_samples=20, centers=3, cluster_std=0.5,
// random_state=0). sklearn yields n_subclusters=10 <= default branching_factor
// (50), so ferrolearn's flat Vec produces the SAME 10 CFs (same centroid set).
// Order may differ (sklearn leaf-order vs ferrolearn insertion-order) — compare
// as a SET (sorted rows). The X below is the exact make_blobs output.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "import numpy as np; from sklearn.cluster import Birch
//     from sklearn.datasets import make_blobs
//     X,_=make_blobs(n_samples=20,centers=3,cluster_std=0.5,random_state=0)
//     b=Birch(n_clusters=3,threshold=0.5).fit(X)
//     print(b.subcluster_centers_.shape[0])  # -> 10
//     print(b.subcluster_centers_.tolist())"
//   ->  10 ; centers (full f64) hardcoded below.
// ===========================================================================

fn blobs20() -> Array2<f64> {
    // Exact make_blobs(n_samples=20, centers=3, cluster_std=0.5,
    // random_state=0) output from the live sklearn 1.5.2 oracle.
    Array2::from_shape_vec(
        (20, 2),
        vec![
            1.048291864126934,
            5.030924080929878,
            -1.7205674219258815,
            2.766730886045455,
            1.1328039293719456,
            3.8767394577975276,
            2.132741234281336,
            1.0867449197390235,
            3.1901448334266815,
            0.17048082263855446,
            -2.051180495755452,
            2.2078732927436353,
            -2.3800391085344117,
            3.8932699589490176,
            1.6113746476178217,
            -0.09273457417402664,
            0.9246606526497161,
            4.509086578417576,
            1.4513142873092897,
            4.228108723299541,
            2.821657128612107,
            1.6323430448880794,
            1.7233096151252982,
            4.201208195565489,
            0.7787726135158386,
            1.2244729576581173,
            2.078146780083601,
            0.80407173492502,
            1.3567889411199918,
            4.364624835694804,
            -1.7817301040977322,
            2.6988451105275297,
            -0.911758672858045,
            3.5190721857253284,
            -1.7008600878849818,
            2.9960567458851126,
            2.487485620862631,
            0.5265811497347159,
            1.1982016949192078,
            4.470624491135523,
        ],
    )
    .unwrap()
}

/// Guard: ferrolearn `subcluster_centers()` on `blobs20` matches sklearn's 10
/// `subcluster_centers_` VALUES (oracle above) as a SET (sorted rows) to ~1e-9.
#[test]
fn green_birch_subcluster_centers_blobs20() {
    // sklearn oracle subcluster_centers_ (full f64), 10 rows.
    let sklearn_centers: [[f64; 2]; 10] = [
        [1.0570514038986192, 4.670211716827659],
        [-1.8135845274160118, 2.6673765088004333],
        [1.4160541932316313, 4.1676703030893405],
        [2.344181714325681, 1.174386566517374],
        [3.1901448334266815, 0.17048082263855446],
        [-2.3800391085344117, 3.8932699589490176],
        [1.6113746476178217, -0.09273457417402664],
        [0.7787726135158386, 1.2244729576581173],
        [-0.911758672858045, 3.5190721857253284],
        [2.487485620862631, 0.5265811497347159],
    ];
    let mut sklearn_rows: Vec<Vec<f64>> = sklearn_centers.iter().map(|r| r.to_vec()).collect();
    sklearn_rows.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let x = blobs20();
    let fitted = Birch::<f64>::new()
        .with_threshold(0.5)
        .with_n_clusters(3)
        .fit(&x, &())
        .unwrap();
    let ferro_rows = sorted_rows(fitted.subcluster_centers());

    assert_eq!(
        ferro_rows.len(),
        10,
        "blobs20 n_subclusters: ferrolearn {} vs sklearn 10",
        ferro_rows.len()
    );
    for (i, (fr, sr)) in ferro_rows.iter().zip(sklearn_rows.iter()).enumerate() {
        for (j, (&f, &s)) in fr.iter().zip(sr.iter()).enumerate() {
            assert!(
                (f - s).abs() <= 1e-9,
                "blobs20 centroid[{i}][{j}] (sorted): ferrolearn {f} vs sklearn \
                 {s} (diff {})",
                (f - s).abs()
            );
        }
    }
}

// ===========================================================================
// GREEN — REQ-10 / REQ-4 bound: threshold=0 and branching_factor=1 rejected.
//
// sklearn _parameter_constraints: threshold Interval(Real,0,None,
// closed="neither") => threshold=0 rejected; branching_factor Interval(Integral,
// 1,None,closed="neither") => >= 2, so branching_factor=1 rejected. Both raise
// InvalidParameterError. ferrolearn `fn fit` rejects threshold<=0 and
// branching_factor<2 with FerroError::InvalidParameter — same boundary.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "import numpy as np; from sklearn.cluster import Birch
//     X=np.array([[0.,0.],[1.,1.],[2.,2.],[3.,3.]])
//     for thr in [0.0]:
//       try: Birch(threshold=thr).fit(X)
//       except Exception as e: print('threshold',thr,type(e).__name__)
//     for bf in [1]:
//       try: Birch(branching_factor=bf).fit(X)
//       except Exception as e: print('bf',bf,type(e).__name__)"
//   ->  threshold 0.0 InvalidParameterError
//       bf 1 InvalidParameterError
// ===========================================================================

/// Guard: ferrolearn `Birch::new().with_threshold(0.0).fit(&X)` returns `Err`
/// (sklearn: `InvalidParameterError`, threshold must be in (0, inf)), and
/// `with_branching_factor(1).fit(&X)` returns `Err` (sklearn: branching_factor
/// >= 2). Both per the oracle above.
#[test]
fn green_birch_threshold_validation() {
    let x = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]).unwrap();

    let r_thr = Birch::<f64>::new().with_threshold(0.0).fit(&x, &());
    assert!(
        r_thr.is_err(),
        "fit with threshold=0.0 should error (sklearn: InvalidParameterError, \
         threshold must be in (0, inf)), got Ok"
    );

    let r_bf = Birch::<f64>::new()
        .with_threshold(0.5)
        .with_branching_factor(1)
        .fit(&x, &());
    assert!(
        r_bf.is_err(),
        "fit with branching_factor=1 should error (sklearn: \
         InvalidParameterError, branching_factor must be >= 2), got Ok"
    );
}
