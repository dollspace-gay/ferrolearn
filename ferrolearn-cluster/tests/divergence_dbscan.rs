//! Value-contract guards + (any) divergence pins for `DBSCAN` /
//! `FittedDBSCAN` (`ferrolearn-cluster/src/dbscan.rs`) against the LIVE
//! scikit-learn 1.5.2 oracle (`from sklearn.cluster import DBSCAN`, mirroring
//! `sklearn/cluster/_dbscan.py`).
//!
//! Every expected value below is a LIVE `sklearn` 1.5.2 oracle value (computed
//! via `python3 -c "..."` run from `/tmp`, quoted above each block) — NEVER
//! copied from the ferrolearn side (goal.md R-CHAR-3).
//!
//! Design doc: `.design/cluster/dbscan.md` (commit e967aae2).
//!
//! ## Why these are GREEN (not RED pins)
//!
//! Unlike the `feature_agglomeration.rs` / `spectral.rs` siblings, DBSCAN's
//! CORE contract genuinely VALUE-MATCHES the live oracle EXACTLY. DBSCAN is
//! deterministic — no RNG, no iterative optimizer, no embedding step. With the
//! Euclidean metric and no `sample_weight`, `labels_` and
//! `core_sample_indices_` are a pure function of the distance graph and the
//! index-ordered cluster expansion (`dbscan_inner`,
//! `sklearn/cluster/_dbscan_inner.pyx`). DBSCAN cluster numbering is
//! deterministic, so labels are asserted ELEMENT-WISE EXACT (NOT
//! up-to-permutation), including the load-bearing shared-border tie-break
//! (first-reaching cluster wins) and the noise (`-1`) set.
//!
//! GREEN guards (all PASS now — they pin the value parity that holds):
//! - `green_dbscan_labels_two_clusters` (REQ-1, Fixture A) — exact `labels_`.
//! - `green_dbscan_labels_three_clusters` (REQ-1, Fixture B) — exact `labels_`.
//! - `green_dbscan_shared_border_tie` (REQ-1/REQ-2, Fixture C) — the load-
//!   bearing tie case: a non-core point reachable from two clusters lands in
//!   the FIRST-reaching cluster, and is EXCLUDED from `core_sample_indices_`.
//! - `green_dbscan_noise_and_core_indices` (REQ-1/REQ-2, Fixture D) — a
//!   noise-heavy `make_blobs` case; `labels_` (incl. `-1`) AND
//!   `core_sample_indices_` exact.
//! - `green_dbscan_eps_zero_rejected` (REQ-3) — `eps=0` rejected (sklearn
//!   `Interval(Real, 0, None, closed="neither")` → `InvalidParameterError`).
//! - `green_dbscan_min_samples_one_self_core` (REQ-2) — `min_samples=1` on a
//!   single point → label 0, core.
//!
//! sample_weight (#947): SHIPPED. `DBSCAN<F>` gains `with_sample_weight` and a
//! weighted core mask `is_core[i] = sum(w[neighbors]) >= min_samples`
//! (`_dbscan.py:427-435`). Pinned green below by
//! `green_dbscan_weighted_*` / `green_dbscan_all_ones_equals_unweighted` /
//! `green_dbscan_fractional_weight_boundary` /
//! `green_dbscan_sample_weight_wrong_length_errs`.
//!
//! ## Documented NOT-STARTED surface (feature ADDITIONS / API changes, NOT
//! minimal single-file fixes — NO forced test):
//!
//! metric / p / metric_params (#948): sklearn accepts any `pairwise_distances`
//! metric (`_dbscan.py:334-337`), `p` for Minkowski (`:341`), default
//! `'euclidean'` (`:350`). Oracle:
//! `DBSCAN(eps=1.2,min_samples=2,metric='manhattan').fit([[0,0],[0.8,0.8],
//! [0.85,0.85]]).labels_` = `[-1,0,0]` vs euclidean `[0,0,0]`. ferrolearn
//! `region_query`/`squared_euclidean` is Euclidean-ONLY — no `metric`/`p`
//! param. Feature addition. Documented only.
//!
//! algorithm / leaf_size / n_jobs (#949): sklearn routes neighbor search
//! through `NearestNeighbors(algorithm, leaf_size, n_jobs)`
//! (`_dbscan.py:411-422`). ferrolearn uses a fixed brute-force `O(n^2)`
//! `region_query` with no parameter. (Brute force value-matches the default
//! `'auto'`; the divergence is the absent parameter surface.) Documented only.
//!
//! components_ attr (#950): sklearn `components_ = X[core_sample_indices_]
//! .copy()` (`_dbscan.py:441-446`). `FittedDBSCAN` exposes
//! `core_sample_indices()` but has NO `components_` accessor and does not
//! retain `X`. Missing surface. Documented only.
//!
//! eps=0.5 default + error ABI (#946): sklearn `__init__` `eps=0.5` default
//! (`_dbscan.py:347`); ferrolearn `fn new(eps: F)` REQUIRES `eps` (no default).
//! Also the validation error TYPE is `FerroError::InvalidParameter`, not
//! sklearn's `InvalidParameterError`/`ValueError` ABI. Documented only.
//!
//! ferray substrate (#951): `dbscan.rs` imports `ndarray::{Array1, Array2}` +
//! `num_traits::Float`, not `ferray-core`. Not migrated. Documented only.

use ferrolearn_cluster::DBSCAN;
use ferrolearn_cluster::dbscan::DbscanMetric;
use ferrolearn_core::traits::Fit;
use ndarray::{Array1, Array2};

// ===========================================================================
// GREEN — Fixture A: two well-separated 4-point squares, exact labels_.
//
// sklearn `DBSCAN(eps=1.5, min_samples=2).fit(X).labels_` numbers clusters by
// the first unlabeled core in index order (`dbscan_inner`,
// `sklearn/cluster/_dbscan_inner.pyx`; `_dbscan.py:431-439`). ferrolearn
// `DBSCAN::new(1.5).with_min_samples(2).fit_predict(&X)` mirrors this. DBSCAN
// numbering is deterministic — EXACT, not up-to-permutation.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "import numpy as np; from sklearn.cluster import DBSCAN
//     A=np.array([[0.,0.],[0.5,0.],[0.,0.5],[0.5,0.5],
//                 [10.,10.],[10.5,10.],[10.,10.5],[10.5,10.5]])
//     m=DBSCAN(eps=1.5,min_samples=2).fit(A)
//     print(m.labels_.tolist(), m.core_sample_indices_.tolist())"
//   ->  [0, 0, 0, 0, 1, 1, 1, 1]   [0, 1, 2, 3, 4, 5, 6, 7]
// ===========================================================================

fn fixture_a() -> Array2<f64> {
    Array2::from_shape_vec(
        (8, 2),
        vec![
            0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 10.0, 10.0, 10.5, 10.0, 10.0, 10.5, 10.5, 10.5,
        ],
    )
    .unwrap()
}

/// Guard: ferrolearn `DBSCAN::new(1.5).with_min_samples(2).fit_predict(&A)`
/// equals sklearn `DBSCAN(eps=1.5, min_samples=2).fit(A).labels_` =
/// `[0,0,0,0,1,1,1,1]` (oracle above) EXACTLY, element-wise.
#[test]
fn green_dbscan_labels_two_clusters() {
    // sklearn oracle labels (Fixture A).
    let sklearn_labels: [isize; 8] = [0, 0, 0, 0, 1, 1, 1, 1];

    let x = fixture_a();
    let labels = DBSCAN::<f64>::new(1.5)
        .with_min_samples(2)
        .fit_predict(&x)
        .unwrap();

    assert_eq!(labels.len(), 8);
    for (i, &exp) in sklearn_labels.iter().enumerate() {
        assert_eq!(
            labels[i], exp,
            "Fixture A label[{i}]: ferrolearn {} vs sklearn {exp} (DBSCAN \
             numbering is deterministic — exact match required)",
            labels[i]
        );
    }
}

// ===========================================================================
// GREEN — Fixture B: three 3-point blobs, exact labels_.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "import numpy as np; from sklearn.cluster import DBSCAN
//     B=np.array([[0.,0.],[0.1,0.],[0.,0.1],[5.,5.],[5.1,5.],[5.,5.1],
//                 [10.,0.],[10.1,0.],[10.,0.1]])
//     m=DBSCAN(eps=0.5,min_samples=2).fit(B)
//     print(m.labels_.tolist(), m.core_sample_indices_.tolist())"
//   ->  [0, 0, 0, 1, 1, 1, 2, 2, 2]   [0, 1, 2, 3, 4, 5, 6, 7, 8]
// ===========================================================================

fn fixture_b() -> Array2<f64> {
    Array2::from_shape_vec(
        (9, 2),
        vec![
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1, 10.0, 0.0, 10.1, 0.0, 10.0,
            0.1,
        ],
    )
    .unwrap()
}

/// Guard: ferrolearn `DBSCAN::new(0.5).with_min_samples(2).fit_predict(&B)`
/// equals sklearn `DBSCAN(eps=0.5, min_samples=2).fit(B).labels_` =
/// `[0,0,0,1,1,1,2,2,2]` (oracle above) EXACTLY, element-wise.
#[test]
fn green_dbscan_labels_three_clusters() {
    // sklearn oracle labels (Fixture B).
    let sklearn_labels: [isize; 9] = [0, 0, 0, 1, 1, 1, 2, 2, 2];

    let x = fixture_b();
    let labels = DBSCAN::<f64>::new(0.5)
        .with_min_samples(2)
        .fit_predict(&x)
        .unwrap();

    assert_eq!(labels.len(), 9);
    for (i, &exp) in sklearn_labels.iter().enumerate() {
        assert_eq!(
            labels[i], exp,
            "Fixture B label[{i}]: ferrolearn {} vs sklearn {exp} (exact match \
             required)",
            labels[i]
        );
    }
}

// ===========================================================================
// GREEN — Fixture C: SHARED-BORDER tie-break (the load-bearing case).
//
// A left core (0,0) with three satellites {(-0.5,0),(0,0.5),(0,-0.5)}, a right
// core (2,0) with three satellites {(2.5,0),(2,0.5),(2,-0.5)}, and a single
// BORDER point (1,0) at idx 4 that is reachable (dist 1.0 <= eps) from the left
// core (idx 0) AND the right core (idx 5), but is NOT itself core (only 3
// neighbors < min_samples=4). The border point must be assigned to the
// FIRST-reaching cluster in index order (cluster 0, expanded first from the
// lowest unlabeled core idx 0), matching `dbscan_inner`'s "skip if already
// labeled" rule (`_dbscan_inner.pyx`; ferrolearn `fn fit` guards
// `if labels[neighbor] == -1`). idx 4 must be EXCLUDED from
// `core_sample_indices_`.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "import numpy as np; from sklearn.cluster import DBSCAN
//     C=np.array([[0.,0.],[-0.5,0.],[0.,0.5],[0.,-0.5],[1.,0.],
//                 [2.,0.],[2.5,0.],[2.,0.5],[2.,-0.5]])
//     m=DBSCAN(eps=1.0,min_samples=4).fit(C)
//     print(m.labels_.tolist(), m.core_sample_indices_.tolist())"
//   ->  [0, 0, 0, 0, 0, 1, 1, 1, 1]   [0, 1, 2, 3, 5, 6, 7, 8]
// ===========================================================================

fn fixture_c() -> Array2<f64> {
    Array2::from_shape_vec(
        (9, 2),
        vec![
            0.0, 0.0, -0.5, 0.0, 0.0, 0.5, 0.0, -0.5, 1.0, 0.0, 2.0, 0.0, 2.5, 0.0, 2.0, 0.5, 2.0,
            -0.5,
        ],
    )
    .unwrap()
}

/// Guard: on Fixture C, ferrolearn `labels` == sklearn `[0,0,0,0,0,1,1,1,1]`
/// (oracle above) EXACTLY — the border point idx 4 lands in cluster 0 (the
/// FIRST-reaching cluster). AND `core_sample_indices()` == sklearn
/// `[0,1,2,3,5,6,7,8]` EXACTLY — idx 4 (the border point) is EXCLUDED.
#[test]
fn green_dbscan_shared_border_tie() {
    // sklearn oracle (Fixture C).
    let sklearn_labels: [isize; 9] = [0, 0, 0, 0, 0, 1, 1, 1, 1];
    let sklearn_core: [usize; 8] = [0, 1, 2, 3, 5, 6, 7, 8];

    let x = fixture_c();
    let fitted = DBSCAN::<f64>::new(1.0)
        .with_min_samples(4)
        .fit(&x, &())
        .unwrap();
    let labels = fitted.labels();

    for (i, &exp) in sklearn_labels.iter().enumerate() {
        assert_eq!(
            labels[i], exp,
            "Fixture C label[{i}]: ferrolearn {} vs sklearn {exp} (shared-border \
             tie: border idx4 must join the FIRST-reaching cluster 0)",
            labels[i]
        );
    }

    assert_eq!(
        fitted.core_sample_indices(),
        &sklearn_core,
        "Fixture C core_sample_indices_: ferrolearn {:?} vs sklearn {:?} \
         (border idx4 must be excluded)",
        fitted.core_sample_indices(),
        sklearn_core
    );
}

// ===========================================================================
// GREEN — Fixture D: make_blobs(n_samples=20, centers=3, cluster_std=0.4,
// random_state=42), eps=0.5, min_samples=3 — a noise-heavy case (11 noise
// points). Both `labels_` (incl. -1 noise) and `core_sample_indices_` exact.
//
// The X coordinates below are the full-precision repr of the sklearn
// make_blobs output (so the Rust fixture reproduces sklearn's X exactly).
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "import numpy as np; from sklearn.cluster import DBSCAN
//     from sklearn.datasets import make_blobs
//     X,_=make_blobs(n_samples=20,centers=3,cluster_std=0.4,random_state=42)
//     m=DBSCAN(eps=0.5,min_samples=3).fit(X)
//     print(m.labels_.tolist()); print(m.core_sample_indices_.tolist())"
//   ->  labels [0,0,0,-1,1,-1,0,1,1,1,-1,-1,-1,0,-1,-1,-1,-1,1,-1]
//       core   [0, 1, 2, 4, 6, 7, 8, 9, 13]
// ===========================================================================

fn fixture_d() -> Array2<f64> {
    // Full-precision repr of make_blobs(n_samples=20, centers=3,
    // cluster_std=0.4, random_state=42) from the live sklearn 1.5.2 oracle.
    Array2::from_shape_vec(
        (20, 2),
        vec![
            -2.6969873774267312,
            9.231310145632708,
            -3.1991647560579635,
            8.789371116501933,
            -2.91433007118652,
            9.13998506123643,
            -2.4124127144263365,
            8.248974030335201,
            4.17948140525918,
            2.1234488912790006,
            -7.410901610710642,
            -6.801365098928297,
            -2.694564700177735,
            8.827994226770219,
            4.634479946332927,
            1.5500853123583718,
            4.422125746418028,
            2.0175387198246786,
            4.39962336026058,
            1.8564921840234212,
            -6.796081753149368,
            -7.663977642827857,
            5.226138343796723,
            1.8828591637461178,
            -6.5842405591531055,
            -6.811562280799959,
            -2.8724072532612346,
            8.449364647664204,
            -6.9258865041065665,
            -7.000551071511662,
            -1.8775124968497936,
            9.321260019859485,
            -7.471035987298241,
            -7.16804727663383,
            -6.550609226309994,
            -7.368447053264355,
            4.666890118103271,
            1.4032704094553492,
            4.399196191336342,
            2.714080957744307,
        ],
    )
    .unwrap()
}

/// Guard: on Fixture D (make_blobs, 11 noise points), ferrolearn `labels()`
/// (incl. -1) == sklearn `[0,0,0,-1,1,-1,0,1,1,1,-1,-1,-1,0,-1,-1,-1,-1,1,-1]`
/// EXACTLY, AND `core_sample_indices()` == sklearn `[0,1,2,4,6,7,8,9,13]`
/// EXACTLY (oracle above).
#[test]
fn green_dbscan_noise_and_core_indices() {
    // sklearn oracle (Fixture D).
    let sklearn_labels: [isize; 20] = [
        0, 0, 0, -1, 1, -1, 0, 1, 1, 1, -1, -1, -1, 0, -1, -1, -1, -1, 1, -1,
    ];
    let sklearn_core: [usize; 9] = [0, 1, 2, 4, 6, 7, 8, 9, 13];

    let x = fixture_d();
    let fitted = DBSCAN::<f64>::new(0.5)
        .with_min_samples(3)
        .fit(&x, &())
        .unwrap();
    let labels = fitted.labels();

    for (i, &exp) in sklearn_labels.iter().enumerate() {
        assert_eq!(
            labels[i], exp,
            "Fixture D label[{i}]: ferrolearn {} vs sklearn {exp} (noise-heavy \
             blobs; -1 = noise; exact match required)",
            labels[i]
        );
    }

    assert_eq!(
        fitted.core_sample_indices(),
        &sklearn_core,
        "Fixture D core_sample_indices_: ferrolearn {:?} vs sklearn {:?}",
        fitted.core_sample_indices(),
        sklearn_core
    );
}

// ===========================================================================
// GREEN — eps=0 is REJECTED at fit (both sides reject).
//
// sklearn `_parameter_constraints["eps"] = Interval(Real, 0.0, None,
// closed="neither")` (`_dbscan.py:332`) → eps=0 is OUTSIDE (0, inf) and raises
// `InvalidParameterError`. ferrolearn `fn fit` rejects `self.eps <= F::zero()`
// with `FerroError::InvalidParameter`. Both error at the same boundary.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "import numpy as np; from sklearn.cluster import DBSCAN
//     try: DBSCAN(eps=0.0).fit(np.array([[0.,0.],[1.,1.]]))
//     except Exception as e: print(type(e).__name__)"
//   ->  InvalidParameterError
// ===========================================================================

/// Guard: ferrolearn `DBSCAN::new(0.0).fit(&X, &())` returns `Err`, matching
/// sklearn which raises `InvalidParameterError` for `eps=0` (outside
/// `Interval(Real, 0, None, closed="neither")`, `_dbscan.py:332`).
#[test]
fn green_dbscan_eps_zero_rejected() {
    let x = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap();
    let result = DBSCAN::<f64>::new(0.0).fit(&x, &());

    assert!(
        result.is_err(),
        "fit with eps=0.0 should error (sklearn: InvalidParameterError, eps must \
         be in (0, inf)), got Ok"
    );
}

// ===========================================================================
// GREEN — min_samples=1 on a single point: self-core, label 0.
//
// With min_samples=1, a point's own neighborhood (which includes itself) has 1
// member >= 1, so it is a core point and forms its own cluster. sklearn neighbor
// search leaves the point itself in (`_dbscan.py:400-402`); ferrolearn
// `region_query` includes self. Both → label [0], core [0].
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "import numpy as np; from sklearn.cluster import DBSCAN
//     m=DBSCAN(eps=1.0,min_samples=1).fit(np.array([[5.,5.]]))
//     print(m.labels_.tolist(), m.core_sample_indices_.tolist())"
//   ->  [0]   [0]
// ===========================================================================

/// Guard: ferrolearn `DBSCAN::new(1.0).with_min_samples(1)` on a single point
/// → label `[0]` and core `[0]`, matching sklearn (oracle above).
#[test]
fn green_dbscan_min_samples_one_self_core() {
    let x = Array2::from_shape_vec((1, 2), vec![5.0, 5.0]).unwrap();
    let fitted = DBSCAN::<f64>::new(1.0)
        .with_min_samples(1)
        .fit(&x, &())
        .unwrap();

    assert_eq!(
        fitted.labels()[0],
        0,
        "single point with min_samples=1: ferrolearn label {} vs sklearn 0 \
         (self-core forms cluster 0)",
        fitted.labels()[0]
    );
    assert_eq!(
        fitted.core_sample_indices(),
        &[0usize],
        "single point with min_samples=1: ferrolearn core {:?} vs sklearn [0]",
        fitted.core_sample_indices()
    );
}

// ===========================================================================
// GREEN — sample_weight (REQ-5): a high-weight point FLIPS to its own core.
//
// sklearn `fit(X, sample_weight=w)` sets `n_neighbors[i] =
// sum(sample_weight[neighbors])` then `core_samples = n_neighbors >=
// min_samples` (`_dbscan.py:427-435`). With `w0=5 >= min_samples=3`, the
// near-isolated point idx0 becomes core BY ITSELF (its neighborhood is just
// {0}, weight 5 >= 3) and forms a singleton cluster — whereas UNWEIGHTED it is
// noise (1 neighbor < 3).
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "import numpy as np; from sklearn.cluster import DBSCAN
//     X=np.array([[0.,0.],[5.,5.],[5.1,5.]])
//     w=DBSCAN(eps=0.5,min_samples=3).fit(X, sample_weight=[5.,1.,1.])
//     print('w', w.labels_.tolist(), w.core_sample_indices_.tolist())
//     u=DBSCAN(eps=0.5,min_samples=3).fit(X)
//     print('u', u.labels_.tolist(), u.core_sample_indices_.tolist())"
//   ->  w [0, -1, -1]   [0]
//       u [-1, -1, -1]  []
// ===========================================================================

/// Guard: weight `[5,1,1]` makes idx0 its own core/cluster — ferrolearn
/// `labels` == sklearn `[0,-1,-1]`, core == `[0]`; and the unweighted call on
/// the same X is `[-1,-1,-1]` / `[]` (the FLIP, oracle above).
#[test]
fn green_dbscan_weighted_flips_isolated_to_core() {
    // sklearn oracle.
    let sklearn_weighted_labels: [isize; 3] = [0, -1, -1];
    let sklearn_weighted_core: [usize; 1] = [0];
    let sklearn_unweighted_labels: [isize; 3] = [-1, -1, -1];

    let x = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 5.0, 5.0, 5.1, 5.0]).unwrap();

    let weighted = DBSCAN::<f64>::new(0.5)
        .with_min_samples(3)
        .with_sample_weight(Array1::from_vec(vec![5.0, 1.0, 1.0]))
        .fit(&x, &())
        .unwrap();

    for (i, &exp) in sklearn_weighted_labels.iter().enumerate() {
        assert_eq!(
            weighted.labels()[i],
            exp,
            "weighted label[{i}]: ferrolearn {} vs sklearn {exp} (w0=5 >= \
             min_samples=3 makes idx0 self-core)",
            weighted.labels()[i]
        );
    }
    assert_eq!(
        weighted.core_sample_indices(),
        &sklearn_weighted_core,
        "weighted core: ferrolearn {:?} vs sklearn [0]",
        weighted.core_sample_indices()
    );

    // The unweighted call on the SAME X must remain noise (the FLIP).
    let unweighted = DBSCAN::<f64>::new(0.5)
        .with_min_samples(3)
        .fit(&x, &())
        .unwrap();
    for (i, &exp) in sklearn_unweighted_labels.iter().enumerate() {
        assert_eq!(
            unweighted.labels()[i],
            exp,
            "unweighted label[{i}]: ferrolearn {} vs sklearn {exp}",
            unweighted.labels()[i]
        );
    }
    assert!(
        unweighted.core_sample_indices().is_empty(),
        "unweighted core should be empty, got {:?}",
        unweighted.core_sample_indices()
    );
}

// ===========================================================================
// GREEN — sample_weight (REQ-5): small fractional weights DEMOTE a core to
// noise. A dense 4-point cluster (each point's neighborhood = all 4, since
// eps=0.5 covers the [0,0.1] square) is core UNWEIGHTED (4 neighbors >=
// min_samples=4). With `w=[0.5,0.5,0.5,0.5]`, each neighbor-weight-sum =
// 4*0.5 = 2.0 < 4, so EVERY point is demoted to noise.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "import numpy as np; from sklearn.cluster import DBSCAN
//     X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[0.1,0.1]])
//     u=DBSCAN(eps=0.5,min_samples=4).fit(X)
//     print('u', u.labels_.tolist(), u.core_sample_indices_.tolist())
//     d=DBSCAN(eps=0.5,min_samples=4).fit(X, sample_weight=[0.5,0.5,0.5,0.5])
//     print('d', d.labels_.tolist(), d.core_sample_indices_.tolist())"
//   ->  u [0, 0, 0, 0]      [0, 1, 2, 3]
//       d [-1, -1, -1, -1]  []
// ===========================================================================

/// Guard: fractional weight `[0.5;4]` demotes the dense cluster to all-noise —
/// ferrolearn `labels` == sklearn `[-1,-1,-1,-1]`, core == `[]`; unweighted is
/// one cluster `[0,0,0,0]` / core `[0,1,2,3]` (oracle above).
#[test]
fn green_dbscan_weighted_demotes_core_to_noise() {
    let sklearn_demote_labels: [isize; 4] = [-1, -1, -1, -1];
    let sklearn_unweighted_labels: [isize; 4] = [0, 0, 0, 0];
    let sklearn_unweighted_core: [usize; 4] = [0, 1, 2, 3];

    let x = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1]).unwrap();

    let demoted = DBSCAN::<f64>::new(0.5)
        .with_min_samples(4)
        .with_sample_weight(Array1::from_vec(vec![0.5, 0.5, 0.5, 0.5]))
        .fit(&x, &())
        .unwrap();
    for (i, &exp) in sklearn_demote_labels.iter().enumerate() {
        assert_eq!(
            demoted.labels()[i],
            exp,
            "demote label[{i}]: ferrolearn {} vs sklearn {exp} (sum 4*0.5=2.0 < \
             min_samples=4)",
            demoted.labels()[i]
        );
    }
    assert!(
        demoted.core_sample_indices().is_empty(),
        "demote core should be empty, got {:?}",
        demoted.core_sample_indices()
    );

    let unweighted = DBSCAN::<f64>::new(0.5)
        .with_min_samples(4)
        .fit(&x, &())
        .unwrap();
    for (i, &exp) in sklearn_unweighted_labels.iter().enumerate() {
        assert_eq!(unweighted.labels()[i], exp);
    }
    assert_eq!(unweighted.core_sample_indices(), &sklearn_unweighted_core);
}

// ===========================================================================
// GREEN — sample_weight (REQ-5): ALL-ONES reproduces the unweighted result
// EXACTLY (the equivalence guard). `_check_sample_weight(None)` returns all
// ones, so the weighted-sum path with `w=[1;n]` is identical to `len`.
//
// Live oracle (sklearn 1.5.2, run from /tmp) — Fixture-A coords:
//   python3 -c "import numpy as np; from sklearn.cluster import DBSCAN
//     A=np.array([[0.,0.],[0.5,0.],[0.,0.5],[0.5,0.5],
//                 [10.,10.],[10.5,10.],[10.,10.5],[10.5,10.5]])
//     m=DBSCAN(eps=1.5,min_samples=2).fit(A, sample_weight=np.ones(8))
//     print(m.labels_.tolist(), m.core_sample_indices_.tolist())"
//   ->  [0, 0, 0, 0, 1, 1, 1, 1]   [0, 1, 2, 3, 4, 5, 6, 7]
// ===========================================================================

/// Guard: `with_sample_weight([1;8])` on Fixture A == sklearn
/// `[0,0,0,0,1,1,1,1]` / core `[0..8]` (oracle above), AND equals the
/// unweighted ferrolearn result element-wise (the equivalence guard).
#[test]
fn green_dbscan_all_ones_equals_unweighted() {
    let sklearn_labels: [isize; 8] = [0, 0, 0, 0, 1, 1, 1, 1];
    let sklearn_core: [usize; 8] = [0, 1, 2, 3, 4, 5, 6, 7];

    let x = fixture_a();

    let weighted = DBSCAN::<f64>::new(1.5)
        .with_min_samples(2)
        .with_sample_weight(Array1::from_elem(8, 1.0))
        .fit(&x, &())
        .unwrap();
    let unweighted = DBSCAN::<f64>::new(1.5)
        .with_min_samples(2)
        .fit(&x, &())
        .unwrap();

    for (i, &exp) in sklearn_labels.iter().enumerate() {
        assert_eq!(
            weighted.labels()[i],
            exp,
            "all-ones label[{i}]: ferrolearn {} vs sklearn {exp}",
            weighted.labels()[i]
        );
        // Equivalence guard: all-ones == unweighted, element-wise.
        assert_eq!(
            weighted.labels()[i],
            unweighted.labels()[i],
            "all-ones must equal unweighted at [{i}]"
        );
    }
    assert_eq!(weighted.core_sample_indices(), &sklearn_core);
    assert_eq!(
        weighted.core_sample_indices(),
        unweighted.core_sample_indices(),
        "all-ones core must equal unweighted core"
    );
}

// ===========================================================================
// GREEN — sample_weight (REQ-5): the FLOAT `>=` boundary. A dense 8-point
// cluster (eps=1.0 covers it all, so each neighborhood = all 8) with
// `w=[0.5;8]`: each neighbor-weight-sum = 8*0.5 = 4.0. With min_samples=4 that
// is EXACTLY at the `>=` boundary (4.0 >= 4 -> core, one cluster); with
// min_samples=5 it is below (4.0 < 5 -> all noise).
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "import numpy as np; from sklearn.cluster import DBSCAN
//     X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[0.1,0.1],[0.05,0.05],
//                 [0.2,0.0],[0.0,0.2],[0.15,0.1]])
//     a=DBSCAN(eps=1.0,min_samples=4).fit(X, sample_weight=np.full(8,0.5))
//     print('a', a.labels_.tolist(), a.core_sample_indices_.tolist())
//     b=DBSCAN(eps=1.0,min_samples=5).fit(X, sample_weight=np.full(8,0.5))
//     print('b', b.labels_.tolist(), b.core_sample_indices_.tolist())"
//   ->  a [0,0,0,0,0,0,0,0]            [0,1,2,3,4,5,6,7]
//       b [-1,-1,-1,-1,-1,-1,-1,-1]   []
// ===========================================================================

fn fixture_dense8() -> Array2<f64> {
    Array2::from_shape_vec(
        (8, 2),
        vec![
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1, 0.05, 0.05, 0.2, 0.0, 0.0, 0.2, 0.15, 0.1,
        ],
    )
    .unwrap()
}

/// Guard: with `w=[0.5;8]` and eps=1.0 (neighborhood = all 8), the weight-sum
/// 4.0 is core at min_samples=4 (`>=` boundary, sklearn `[0;8]` / core `[0..8]`)
/// and noise at min_samples=5 (sklearn `[-1;8]` / `[]`) — oracle above.
#[test]
fn green_dbscan_fractional_weight_boundary() {
    let x = fixture_dense8();
    let w = || Array1::from_elem(8, 0.5);

    // min_samples=4: 4.0 >= 4 -> all one cluster.
    let at = DBSCAN::<f64>::new(1.0)
        .with_min_samples(4)
        .with_sample_weight(w())
        .fit(&x, &())
        .unwrap();
    for i in 0..8 {
        assert_eq!(
            at.labels()[i],
            0,
            "boundary(min_samples=4) label[{i}]: ferrolearn {} vs sklearn 0 \
             (4.0 >= 4)",
            at.labels()[i]
        );
    }
    assert_eq!(
        at.core_sample_indices(),
        &[0usize, 1, 2, 3, 4, 5, 6, 7],
        "boundary(min_samples=4) core: ferrolearn {:?} vs sklearn [0..8]",
        at.core_sample_indices()
    );

    // min_samples=5: 4.0 < 5 -> all noise.
    let below = DBSCAN::<f64>::new(1.0)
        .with_min_samples(5)
        .with_sample_weight(w())
        .fit(&x, &())
        .unwrap();
    for i in 0..8 {
        assert_eq!(
            below.labels()[i],
            -1,
            "boundary(min_samples=5) label[{i}]: ferrolearn {} vs sklearn -1 \
             (4.0 < 5)",
            below.labels()[i]
        );
    }
    assert!(
        below.core_sample_indices().is_empty(),
        "boundary(min_samples=5) core should be empty, got {:?}",
        below.core_sample_indices()
    );
}

// ===========================================================================
// GREEN — sample_weight (REQ-5): a WRONG-LENGTH weight returns Err (no panic),
// mirroring `_check_sample_weight`'s shape `ValueError`
// (`sklearn/utils/validation.py:2055-2060`).
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "import numpy as np; from sklearn.cluster import DBSCAN
//     X=np.array([[0.,0.],[5.,5.],[5.1,5.]])
//     try: DBSCAN(eps=0.5,min_samples=3).fit(X, sample_weight=[1.,1.])
//     except Exception as e: print(type(e).__name__)"
//   ->  ValueError   (sample_weight.shape == (2,), expected (3,)!)
// ===========================================================================

/// Guard: a length-2 `sample_weight` on a 3-sample X returns `Err` (no panic),
/// matching sklearn which raises `ValueError` for the shape mismatch (oracle
/// above).
#[test]
fn green_dbscan_sample_weight_wrong_length_errs() {
    let x = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 5.0, 5.0, 5.1, 5.0]).unwrap();
    let result = DBSCAN::<f64>::new(0.5)
        .with_min_samples(3)
        .with_sample_weight(Array1::from_vec(vec![1.0, 1.0]))
        .fit(&x, &());

    assert!(
        result.is_err(),
        "wrong-length sample_weight (len 2 for 3 samples) should error \
         (sklearn: ValueError), got Ok"
    );
}

// ===========================================================================
// RED PIN — sample_weight (REQ-5): the float `>=` boundary is SUMMATION-ORDER
// sensitive, and ferrolearn's sequential `fold` (`dbscan.rs` `Fit::fit`:
// `n.iter().fold(F::zero(), |acc,&j| acc + w[j])`) diverges from sklearn's
// `np.sum(sample_weight[neighbors])` (`_dbscan.py:428`), which uses NumPy
// pairwise summation.
//
// Fixture: 10 points packed into a ~0.02-wide square so that with eps=1.0 every
// point's neighborhood = all 10 indices {0..9} (ascending, same order both
// sides — verified via `NearestNeighbors(radius=1.0).radius_neighbors`). All
// weights = 0.1 (NOT exactly representable in binary f64), min_samples=1.
//
// The weight-sum is `0.1` summed ten times:
//   - sklearn `np.sum(full(10, 0.1))` (pairwise) == EXACTLY 1.0
//     (bits 0x3ff0000000000000) -> `1.0 >= 1` -> EVERY point is core ->
//     all in cluster 0.
//   - ferrolearn sequential `acc += 0.1` ten times == 0.9999999999999999
//     (bits 0x3fefffffffffffff) -> `0.999... >= 1.0` is FALSE -> NO core ->
//     EVERY point is noise (-1), core_sample_indices_ == [].
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "import numpy as np; from sklearn.cluster import DBSCAN
//     X=np.array([[0.0,0.0],[0.01,0.0],[0.0,0.01],[0.01,0.01],[0.02,0.0],
//                 [0.0,0.02],[0.02,0.02],[0.01,0.02],[0.02,0.01],[0.015,0.015]])
//     m=DBSCAN(eps=1.0,min_samples=1).fit(X, sample_weight=np.full(10,0.1))
//     print(m.labels_.tolist(), m.core_sample_indices_.tolist())"
//   ->  [0,0,0,0,0,0,0,0,0,0]   [0,1,2,3,4,5,6,7,8,9]
//
// ferrolearn (the divergence): labels_ == [-1;10], core_sample_indices_ == [].
//
// This is a NEW divergence distinct from #952 (the eps-distance-form boundary):
// here ALL neighbor SETS agree exactly (eps=1.0, identical ascending order); the
// only difference is the float SUMMATION ALGORITHM at the integer `>=` boundary.
// ===========================================================================

/// Divergence: ferrolearn's weighted core-mask sum diverges from
/// `sklearn/cluster/_dbscan.py:428` (`np.sum(sample_weight[neighbors])`).
/// Ten weights of 0.1: sklearn's pairwise `np.sum` == 1.0 (>= min_samples=1 ->
/// all core, all cluster 0); ferrolearn's sequential `fold` == 0.9999999999999999
/// (< 1.0 -> all noise). Neighbor SETS are identical (eps=1.0); only the
/// summation order differs. Tracking: #2189.
#[test]
fn divergence_dbscan_weight_sum_summation_order_boundary() {
    // sklearn 1.5.2 oracle (np.sum is pairwise -> EXACTLY 1.0 -> all core).
    let sklearn_labels: [isize; 10] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let sklearn_core: [usize; 10] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

    let x = Array2::from_shape_vec(
        (10, 2),
        vec![
            0.0, 0.0, 0.01, 0.0, 0.0, 0.01, 0.01, 0.01, 0.02, 0.0, 0.0, 0.02, 0.02, 0.02, 0.01,
            0.02, 0.02, 0.01, 0.015, 0.015,
        ],
    )
    .unwrap();

    let fitted = DBSCAN::<f64>::new(1.0)
        .with_min_samples(1)
        .with_sample_weight(Array1::from_elem(10, 0.1))
        .fit(&x, &())
        .unwrap();

    // sklearn: every point is core (sum 1.0 >= 1) and forms cluster 0.
    for (i, &exp) in sklearn_labels.iter().enumerate() {
        assert_eq!(
            fitted.labels()[i],
            exp,
            "label[{i}]: ferrolearn {} vs sklearn {exp} (np.sum(10x0.1)==1.0 -> \
             core; ferrolearn sequential fold==0.9999999999999999 -> noise)",
            fitted.labels()[i]
        );
    }
    assert_eq!(
        fitted.core_sample_indices(),
        &sklearn_core,
        "core_sample_indices_: ferrolearn {:?} vs sklearn {:?} (summation-order \
         flip at the float `>=` boundary)",
        fitted.core_sample_indices(),
        sklearn_core
    );
}

// ===========================================================================
// GREEN — metric (REQ-6): MANHATTAN vs EUCLIDEAN give DIFFERENT labels_.
//
// sklearn routes the neighbor test through `NearestNeighbors(metric=...,
// p=...)` (`_dbscan.py:411-422`): a point `j` is a neighbor of `i` iff
// `dist(i,j) <= eps`. Fixture M: three points on the diagonal
// {(0,0),(0.8,0.8),(0.85,0.85)} with eps=1.2, min_samples=2. The (0,0)->
// (0.8,0.8) distance is `sqrt(2)*0.8 ≈ 1.131 <= 1.2` (euclidean neighbor) but
// `1.6 > 1.2` (manhattan NON-neighbor). So under manhattan idx0 is isolated
// (noise), under euclidean all three are one cluster. None of the relevant
// distances are within a ULP of eps=1.2.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "import numpy as np; from sklearn.cluster import DBSCAN
//     X=np.array([[0.,0.],[0.8,0.8],[0.85,0.85]])
//     for met in ('euclidean','manhattan'):
//       m=DBSCAN(eps=1.2,min_samples=2,metric=met).fit(X)
//       print(met, m.labels_.tolist(), m.core_sample_indices_.tolist())"
//   ->  euclidean [0, 0, 0]   [0, 1, 2]
//       manhattan [-1, 0, 0]  [1, 2]
// ===========================================================================

fn fixture_m() -> Array2<f64> {
    Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 0.8, 0.8, 0.85, 0.85]).unwrap()
}

/// Guard: on Fixture M, ferrolearn `with_metric(Manhattan)` ==
/// sklearn `metric='manhattan'` labels `[-1,0,0]` / core `[1,2]` EXACTLY, and
/// `with_metric(Euclidean)` == sklearn `metric='euclidean'` `[0,0,0]` /
/// `[0,1,2]` EXACTLY (oracle above) — the metric changes the neighbor graph.
#[test]
fn green_dbscan_manhattan_vs_euclidean_labels() {
    // sklearn oracle (Fixture M).
    let sk_manhattan_labels: [isize; 3] = [-1, 0, 0];
    let sk_manhattan_core: [usize; 2] = [1, 2];
    let sk_euclidean_labels: [isize; 3] = [0, 0, 0];
    let sk_euclidean_core: [usize; 3] = [0, 1, 2];

    let x = fixture_m();

    let manhattan = DBSCAN::<f64>::new(1.2)
        .with_min_samples(2)
        .with_metric(DbscanMetric::Manhattan)
        .fit(&x, &())
        .unwrap();
    for (i, &exp) in sk_manhattan_labels.iter().enumerate() {
        assert_eq!(
            manhattan.labels()[i],
            exp,
            "manhattan label[{i}]: ferrolearn {} vs sklearn {exp} (idx0 isolated: \
             L1 dist 1.6 > eps 1.2)",
            manhattan.labels()[i]
        );
    }
    assert_eq!(
        manhattan.core_sample_indices(),
        &sk_manhattan_core,
        "manhattan core: ferrolearn {:?} vs sklearn [1,2]",
        manhattan.core_sample_indices()
    );

    let euclidean = DBSCAN::<f64>::new(1.2)
        .with_min_samples(2)
        .with_metric(DbscanMetric::Euclidean)
        .fit(&x, &())
        .unwrap();
    for (i, &exp) in sk_euclidean_labels.iter().enumerate() {
        assert_eq!(
            euclidean.labels()[i],
            exp,
            "euclidean label[{i}]: ferrolearn {} vs sklearn {exp} (L2 dist 1.131 \
             <= eps 1.2)",
            euclidean.labels()[i]
        );
    }
    assert_eq!(euclidean.core_sample_indices(), &sk_euclidean_core);
}

/// Guard: the default (`metric='euclidean'`) path is UNCHANGED — an explicit
/// `with_metric(Euclidean)` equals the no-metric run element-wise on Fixture M.
#[test]
fn green_dbscan_default_euclidean_equals_no_metric() {
    let x = fixture_m();

    let explicit = DBSCAN::<f64>::new(1.2)
        .with_min_samples(2)
        .with_metric(DbscanMetric::Euclidean)
        .fit(&x, &())
        .unwrap();
    let default = DBSCAN::<f64>::new(1.2)
        .with_min_samples(2)
        .fit(&x, &())
        .unwrap();

    for i in 0..3 {
        assert_eq!(
            explicit.labels()[i],
            default.labels()[i],
            "explicit-Euclidean must equal no-metric at [{i}]"
        );
    }
    assert_eq!(
        explicit.core_sample_indices(),
        default.core_sample_indices(),
        "explicit-Euclidean core must equal no-metric core"
    );
}

// ===========================================================================
// GREEN — minkowski p (REQ-6): p=1 reproduces manhattan, p=2 reproduces
// euclidean, on Fixture M, all matching sklearn `metric='minkowski', p=...`.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "import numpy as np; from sklearn.cluster import DBSCAN
//     X=np.array([[0.,0.],[0.8,0.8],[0.85,0.85]])
//     for p in (1,2):
//       m=DBSCAN(eps=1.2,min_samples=2,metric='minkowski',p=p).fit(X)
//       print('p',p, m.labels_.tolist(), m.core_sample_indices_.tolist())"
//   ->  p 1 [-1, 0, 0]   [1, 2]
//       p 2 [0, 0, 0]    [0, 1, 2]
// ===========================================================================

/// Guard: `with_p(1.0)` (Minkowski p=1) reproduces sklearn `p=1` =
/// `[-1,0,0]`/`[1,2]` AND equals the explicit Manhattan run; `with_p(2.0)`
/// reproduces sklearn `p=2` = `[0,0,0]`/`[0,1,2]` AND equals the explicit
/// Euclidean run (oracle above).
#[test]
fn green_dbscan_minkowski_p1_p2_collapse() {
    let sk_p1_labels: [isize; 3] = [-1, 0, 0];
    let sk_p1_core: [usize; 2] = [1, 2];
    let sk_p2_labels: [isize; 3] = [0, 0, 0];
    let sk_p2_core: [usize; 3] = [0, 1, 2];

    let x = fixture_m();

    let p1 = DBSCAN::<f64>::new(1.2)
        .with_min_samples(2)
        .with_metric(DbscanMetric::Minkowski(1.0))
        .fit(&x, &())
        .unwrap();
    for (i, &exp) in sk_p1_labels.iter().enumerate() {
        assert_eq!(
            p1.labels()[i],
            exp,
            "minkowski p=1 label[{i}]: ferrolearn {} vs sklearn {exp}",
            p1.labels()[i]
        );
    }
    assert_eq!(p1.core_sample_indices(), &sk_p1_core);

    // p=1 must equal the explicit Manhattan run (the collapse).
    let manhattan = DBSCAN::<f64>::new(1.2)
        .with_min_samples(2)
        .with_metric(DbscanMetric::Manhattan)
        .fit(&x, &())
        .unwrap();
    for i in 0..3 {
        assert_eq!(
            p1.labels()[i],
            manhattan.labels()[i],
            "minkowski p=1 must equal Manhattan at [{i}]"
        );
    }

    let p2 = DBSCAN::<f64>::new(1.2)
        .with_min_samples(2)
        .with_metric(DbscanMetric::Minkowski(2.0))
        .fit(&x, &())
        .unwrap();
    for (i, &exp) in sk_p2_labels.iter().enumerate() {
        assert_eq!(
            p2.labels()[i],
            exp,
            "minkowski p=2 label[{i}]: ferrolearn {} vs sklearn {exp}",
            p2.labels()[i]
        );
    }
    assert_eq!(p2.core_sample_indices(), &sk_p2_core);

    // p=2 must equal the explicit Euclidean run (and hence the default path).
    let euclidean = DBSCAN::<f64>::new(1.2)
        .with_min_samples(2)
        .fit(&x, &())
        .unwrap();
    for i in 0..3 {
        assert_eq!(
            p2.labels()[i],
            euclidean.labels()[i],
            "minkowski p=2 must equal Euclidean at [{i}]"
        );
    }
    assert_eq!(p2.core_sample_indices(), euclidean.core_sample_indices());
}

// ===========================================================================
// GREEN — chebyshev / minkowski p=3 (REQ-6): a fixture where these DIFFER from
// euclidean/manhattan. Fixture N: four points on the diagonal
// {(0,0),(1,1),(1.5,1.5),(2.5,2.5)} with eps=1.3, min_samples=2.
//
// Pairwise (0,0)->(1,1): L2 1.414, L1 2.0, Linf 1.0, Mink-p3 1.260; (1,1)->
// (1.5,1.5): L2 0.707, L1 1.0, Linf 0.5; (1.5,1.5)->(2.5,2.5): L2 1.414, L1
// 2.0, Linf 1.0, Mink-p3 1.260. With eps=1.3:
//   - euclidean/manhattan/mink-p1: idx0 and idx3 are isolated (their nearest L2
//     neighbor 1.414 > 1.3) -> labels [-1,0,0,-1], core [1,2].
//   - chebyshev/mink-p3: all chained (Linf 1.0 <= 1.3, Mink-p3 1.260 <= 1.3) ->
//     labels [0,0,0,0], core [0,1,2,3].
// No distance is within a ULP of eps=1.3.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "import numpy as np; from sklearn.cluster import DBSCAN
//     X=np.array([[0.,0.],[1.,1.],[1.5,1.5],[2.5,2.5]])
//     for met,p in (('euclidean',None),('chebyshev',None),('minkowski',3)):
//       kw=dict(eps=1.3,min_samples=2,metric=met)
//       if p is not None: kw['p']=p
//       m=DBSCAN(**kw).fit(X)
//       print(met,p, m.labels_.tolist(), m.core_sample_indices_.tolist())"
//   ->  euclidean None [-1, 0, 0, -1]   [1, 2]
//       chebyshev None [0, 0, 0, 0]     [0, 1, 2, 3]
//       minkowski 3    [0, 0, 0, 0]     [0, 1, 2, 3]
// ===========================================================================

fn fixture_n() -> Array2<f64> {
    Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 1.0, 1.5, 1.5, 2.5, 2.5]).unwrap()
}

/// Guard: on Fixture N, `with_metric(Chebyshev)` == sklearn `chebyshev`
/// `[0,0,0,0]`/`[0,1,2,3]`, `with_p(3.0)` (Minkowski p=3) == sklearn
/// `minkowski p=3` `[0,0,0,0]`/`[0,1,2,3]`, and the default Euclidean ==
/// sklearn `euclidean` `[-1,0,0,-1]`/`[1,2]` — all EXACT (oracle above).
#[test]
fn green_dbscan_chebyshev_and_minkowski_p3() {
    let sk_euclidean_labels: [isize; 4] = [-1, 0, 0, -1];
    let sk_euclidean_core: [usize; 2] = [1, 2];
    let sk_other_labels: [isize; 4] = [0, 0, 0, 0];
    let sk_other_core: [usize; 4] = [0, 1, 2, 3];

    let x = fixture_n();

    // Euclidean baseline (default path) — idx0/idx3 isolated.
    let euclidean = DBSCAN::<f64>::new(1.3)
        .with_min_samples(2)
        .fit(&x, &())
        .unwrap();
    for (i, &exp) in sk_euclidean_labels.iter().enumerate() {
        assert_eq!(
            euclidean.labels()[i],
            exp,
            "euclidean label[{i}]: ferrolearn {} vs sklearn {exp}",
            euclidean.labels()[i]
        );
    }
    assert_eq!(euclidean.core_sample_indices(), &sk_euclidean_core);

    // Chebyshev — all chained.
    let chebyshev = DBSCAN::<f64>::new(1.3)
        .with_min_samples(2)
        .with_metric(DbscanMetric::Chebyshev)
        .fit(&x, &())
        .unwrap();
    for (i, &exp) in sk_other_labels.iter().enumerate() {
        assert_eq!(
            chebyshev.labels()[i],
            exp,
            "chebyshev label[{i}]: ferrolearn {} vs sklearn {exp}",
            chebyshev.labels()[i]
        );
    }
    assert_eq!(chebyshev.core_sample_indices(), &sk_other_core);

    // Minkowski p=3 — all chained (matches sklearn).
    let p3 = DBSCAN::<f64>::new(1.3)
        .with_min_samples(2)
        .with_metric(DbscanMetric::Minkowski(3.0))
        .fit(&x, &())
        .unwrap();
    for (i, &exp) in sk_other_labels.iter().enumerate() {
        assert_eq!(
            p3.labels()[i],
            exp,
            "minkowski p=3 label[{i}]: ferrolearn {} vs sklearn {exp}",
            p3.labels()[i]
        );
    }
    assert_eq!(p3.core_sample_indices(), &sk_other_core);
}

// ===========================================================================
// GREEN — p validation (REQ-6): a non-positive Minkowski `p` is REJECTED (no
// panic), mirroring sklearn which raises `InvalidParameterError` because
// `NearestNeighbors` requires `p` in `(0, inf]`.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "import numpy as np; from sklearn.cluster import DBSCAN
//     X=np.array([[0.,0.],[1.,1.],[2.,2.]])
//     for p in (-1.0, 0.0):
//       try: DBSCAN(eps=1.5,min_samples=2,metric='minkowski',p=p).fit(X)
//       except Exception as e: print(p, type(e).__name__)"
//   ->  -1.0 InvalidParameterError
//        0.0 InvalidParameterError
// ===========================================================================

/// Guard: `with_p(p)` for `p <= 0` returns `Err` (no panic), matching sklearn
/// which raises `InvalidParameterError` (p must be in `(0, inf]`, oracle above).
#[test]
fn green_dbscan_minkowski_nonpositive_p_rejected() {
    let x = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]).unwrap();

    for &p in &[-1.0f64, 0.0] {
        let result = DBSCAN::<f64>::new(1.5)
            .with_min_samples(2)
            .with_metric(DbscanMetric::Minkowski(p))
            .fit(&x, &());
        assert!(
            result.is_err(),
            "minkowski p={p} should error (sklearn: InvalidParameterError, p in \
             (0, inf]), got Ok"
        );
    }
}

// ===========================================================================
// GREEN guard (was a RED pin, fixed #2192) — metric/p precedence (REQ-6,
// #948): sklearn's `p` is an INDEPENDENT parameter that is IGNORED for
// non-Minkowski metrics. `DBSCAN::with_p` now updates the order ONLY when the
// metric is already `Minkowski` (a no-op otherwise), so
// `.with_metric(Euclidean).with_p(3.0)` stays Euclidean — matching sklearn's
// `DBSCAN(metric='euclidean', p=3)`, which ignores `p`. To select Minkowski use
// `.with_metric(DbscanMetric::Minkowski(p))`.
//
// sklearn `NearestNeighbors(metric=self.metric, ..., p=self.p)` passes `p`
// ALONGSIDE `metric` (`_dbscan.py:411-418`), but `p` only feeds the Minkowski
// metric; for `metric='euclidean'` the L2 metric ignores `p`
// (`_parameter_constraints` accepts `p` for any metric, `_dbscan.py:341`).
//
// Fixture N reused (diagonal {(0,0),(1,1),(1.5,1.5),(2.5,2.5)}, eps=1.3,
// min_samples=2). Euclidean isolates idx0/idx3 (L2 1.414 > 1.3); Minkowski p=3
// chains all four (mink-p3 1.260 <= 1.3). No distance within a ULP of eps=1.3.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "import numpy as np; from sklearn.cluster import DBSCAN
//     X=np.array([[0.,0.],[1.,1.],[1.5,1.5],[2.5,2.5]])
//     m=DBSCAN(eps=1.3,min_samples=2,metric='euclidean',p=3).fit(X)
//     print(m.labels_.tolist(), m.core_sample_indices_.tolist())"
//   ->  [-1, 0, 0, -1]   [1, 2]
// (i.e. IDENTICAL to metric='euclidean' with p=None — p is ignored.)
//
// ferrolearn `with_metric(Euclidean).with_p(3.0)` -> Minkowski(3) ->
// labels [0,0,0,0] / core [0,1,2,3] (the Minkowski-p3 result), diverging.
// ===========================================================================

/// Fixed (#2192): `DBSCAN::with_p` now updates the order ONLY when the metric
/// is already `Minkowski` (a no-op otherwise), mirroring sklearn where `p` is
/// IGNORED for non-Minkowski metrics (`sklearn/cluster/_dbscan.py:341,411-418`).
/// So `.with_metric(Euclidean).with_p(3.0)` stays Euclidean and yields the
/// euclidean result `[-1,0,0,-1]` / core `[1,2]`, matching `DBSCAN(metric=
/// 'euclidean', p=3)`. Tracking: #2192.
#[test]
fn divergence_dbscan_p_ignored_for_euclidean() {
    // sklearn oracle: metric='euclidean', p=3 == euclidean (p IGNORED).
    let sklearn_labels: [isize; 4] = [-1, 0, 0, -1];
    let sklearn_core: [usize; 2] = [1, 2];

    let x = fixture_n();

    // The ferrolearn analogue of sklearn `DBSCAN(metric='euclidean', p=3)`:
    // request the Euclidean metric AND set p=3. sklearn ignores p here.
    let fitted = DBSCAN::<f64>::new(1.3)
        .with_min_samples(2)
        .with_metric(DbscanMetric::Euclidean)
        .with_p(3.0)
        .fit(&x, &())
        .unwrap();

    for (i, &exp) in sklearn_labels.iter().enumerate() {
        assert_eq!(
            fitted.labels()[i],
            exp,
            "metric=euclidean,p=3 label[{i}]: ferrolearn {} vs sklearn {exp} \
             (sklearn IGNORES p for euclidean; ferrolearn with_p overwrites \
             metric to Minkowski(3))",
            fitted.labels()[i]
        );
    }
    assert_eq!(
        fitted.core_sample_indices(),
        &sklearn_core,
        "metric=euclidean,p=3 core: ferrolearn {:?} vs sklearn [1,2] \
         (p must be ignored for euclidean)",
        fitted.core_sample_indices()
    );
}
