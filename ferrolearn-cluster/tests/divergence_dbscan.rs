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
//! ## Documented NOT-STARTED surface (feature ADDITIONS / API changes, NOT
//! minimal single-file fixes — NO forced test):
//!
//! sample_weight (#947): sklearn `fit(X, sample_weight=w)` computes
//! `n_neighbors = sum(sample_weight[neighbors])` (`_dbscan.py:427-429`), so a
//! high-weight point becomes core with FEWER neighbors, CHANGING labels.
//! Oracle: `DBSCAN(eps=0.5,min_samples=3).fit([[0,0],[5,5],[5.1,5]],
//! sample_weight=[5,1,1]).labels_` = `[0,-1,-1]` vs unweighted `[-1,-1,-1]`.
//! ferrolearn `Fit<Array2<F>, ()>` has the unit `()` target — no weight param.
//! Adding it is an API change (a new `Fit` target / weighted-fit method), NOT a
//! minimal single-file fix. Documented only.
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
use ferrolearn_core::traits::Fit;
use ndarray::Array2;

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
