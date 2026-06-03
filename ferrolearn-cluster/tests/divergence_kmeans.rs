//! Divergence + green-guard tests for `ferrolearn-cluster::KMeans` against the
//! live scikit-learn 1.5.2 oracle (`sklearn/cluster/_kmeans.py`).
//!
//! Two groups:
//!
//! **(A) GREEN-GUARDS** — PASS against current code. They pin the SHIPPED
//! Rust-level contracts (design doc `.design/cluster/kmeans.md` REQ-1/2/3),
//! using expected values computed by the live sklearn 1.5.2 oracle (R-CHAR-3)
//! and canonicalized to ignore the label-integer permutation (REQ-9 diverges).
//!
//! **(B) PINS** — FAIL against current code, go green after the owning fix:
//!   - REQ-14 (#1045): `KMeans::new` defaults `n_init = 1` (sklearn `n_init="auto"`
//!     resolves to 1 for `init="k-means++"`, `_kmeans.py:886-888`). ferrolearn
//!     `fn new` currently defaults `10`.
//!   - REQ-6  (#1037): `fit(X).predict(X) == labels_` — sklearn re-runs the E-step
//!     post-loop so labels match the final centers (`_kmeans.py:605-625`).
//!     ferrolearn stores `labels_` against the PRE-swap centers while
//!     `cluster_centers_` is POST-swap, so `predict(X)` can disagree.
//!
//! All sklearn expected values were produced from the installed sklearn 1.5.2
//! oracle (run from /tmp), never literal-copied from the ferrolearn side.

use ferrolearn_cluster::KMeans;
use ferrolearn_core::{Fit, Predict, Transform};
use ndarray::Array2;

/// Canonicalize a label vector to a permutation-invariant partition signature:
/// each distinct label is renamed to the index of its first occurrence. Two
/// label vectors describe the same PARTITION iff their canonical forms are
/// equal. This lets us compare ferrolearn's partition against sklearn's without
/// depending on the (RNG-dependent) absolute integers (REQ-9 diverges).
fn canonicalize(labels: &[usize]) -> Vec<usize> {
    let mut map = std::collections::HashMap::new();
    let mut next = 0usize;
    labels
        .iter()
        .map(|&l| {
            *map.entry(l).or_insert_with(|| {
                let v = next;
                next += 1;
                v
            })
        })
        .collect()
}

// ----------------------------------------------------------------------------
// (A) GREEN-GUARDS — SHIPPED contracts, must PASS against current code.
// ----------------------------------------------------------------------------

/// REQ-1 green-guard (2-blob partition up-to-permutation).
///
/// Fresh separable fixture (NOT the in-tree `make_blobs`): two tight blobs near
/// (2,3) and (20,25). Live oracle:
/// ```text
/// python3 -c "import numpy as np; from sklearn.cluster import KMeans; \
///   X=np.array([[2.,3.],[2.2,2.9],[1.8,3.1],[20.,25.],[20.3,24.8],[19.7,25.2]]); \
///   print(KMeans(n_clusters=2, n_init=5, random_state=0).fit(X).labels_.tolist())"
/// # [1, 1, 1, 0, 0, 0]   -> canonical [0,0,0,1,1,1]
/// ```
#[test]
fn green_req1_two_blob_partition() {
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![
            2.0, 3.0, 2.2, 2.9, 1.8, 3.1, 20.0, 25.0, 20.3, 24.8, 19.7, 25.2,
        ],
    )
    .unwrap();

    // sklearn KMeans(n_clusters=2, n_init=5, random_state=0).fit(X).labels_
    // canonicalized -> [0,0,0,1,1,1] (live oracle).
    let sklearn_canon: Vec<usize> = canonicalize(&[1, 1, 1, 0, 0, 0]);

    let model = KMeans::<f64>::new(2).with_random_state(0).with_n_init(5);
    let fitted = model.fit(&x, &()).unwrap();
    let ferro_canon = canonicalize(&fitted.labels().to_vec());

    assert_eq!(
        ferro_canon, sklearn_canon,
        "ferrolearn KMeans must recover sklearn's 2-blob partition (up to permutation)"
    );
}

/// REQ-1 green-guard (3-blob partition up-to-permutation).
///
/// Fresh separable fixture (NOT in-tree): blobs near (1,1), (50,1), (25,40).
/// Live oracle:
/// ```text
/// python3 -c "import numpy as np; from sklearn.cluster import KMeans; \
///   X=np.array([[1.,1.],[1.3,0.8],[0.7,1.2],[50.,1.],[50.2,1.3],[49.8,0.7],[25.,40.],[25.4,39.6],[24.6,40.3]]); \
///   print(KMeans(n_clusters=3, n_init=5, random_state=0).fit(X).labels_.tolist())"
/// # [2, 2, 2, 0, 0, 0, 1, 1, 1]   -> canonical [0,0,0,1,1,1,2,2,2]
/// ```
#[test]
fn green_req1_three_blob_partition() {
    let x = Array2::from_shape_vec(
        (9, 2),
        vec![
            1.0, 1.0, 1.3, 0.8, 0.7, 1.2, 50.0, 1.0, 50.2, 1.3, 49.8, 0.7, 25.0, 40.0, 25.4, 39.6,
            24.6, 40.3,
        ],
    )
    .unwrap();

    // sklearn KMeans(n_clusters=3, n_init=5, random_state=0).fit(X).labels_
    // canonicalized -> [0,0,0,1,1,1,2,2,2] (live oracle).
    let sklearn_canon: Vec<usize> = canonicalize(&[2, 2, 2, 0, 0, 0, 1, 1, 1]);

    let model = KMeans::<f64>::new(3).with_random_state(0).with_n_init(5);
    let fitted = model.fit(&x, &()).unwrap();
    let ferro_canon = canonicalize(&fitted.labels().to_vec());

    assert_eq!(
        ferro_canon, sklearn_canon,
        "ferrolearn KMeans must recover sklearn's 3-blob partition (up to permutation)"
    );
}

/// REQ-2 green-guard: `predict` is argmin over `transform` distances.
///
/// sklearn `KMeans.predict` assigns each sample to the argmin-distance center
/// and `transform(X).argmin(axis=1) == predict(X)` (`_kmeans.py:1072-1154`;
/// live Probe C: `np.array_equal(T.argmin(1), m.predict(X)) -> True`). This is a
/// self-consistency contract between the two SHIPPED methods — no value parity
/// needed, so it holds for ferrolearn's own (divergent-value) centers.
#[test]
fn green_req2_predict_is_transform_argmin() {
    let x = Array2::from_shape_vec(
        (9, 2),
        vec![
            1.0, 1.0, 1.3, 0.8, 0.7, 1.2, 50.0, 1.0, 50.2, 1.3, 49.8, 0.7, 25.0, 40.0, 25.4, 39.6,
            24.6, 40.3,
        ],
    )
    .unwrap();

    let model = KMeans::<f64>::new(3).with_random_state(0).with_n_init(5);
    let fitted = model.fit(&x, &()).unwrap();

    let dists = fitted.transform(&x).unwrap();
    let pred = fitted.predict(&x).unwrap();

    let k = dists.ncols();
    for i in 0..dists.nrows() {
        // argmin over column j of distances[i, j]
        let mut best_j = 0usize;
        let mut best = f64::INFINITY;
        for j in 0..k {
            if dists[[i, j]] < best {
                best = dists[[i, j]];
                best_j = j;
            }
        }
        assert_eq!(
            best_j, pred[i],
            "transform(X).argmin(axis=1) must equal predict(X) at row {i} (sklearn Probe C True)"
        );
    }
}

/// REQ-3 green-guard: `transform` shape `(n_samples, n_clusters)`, all entries
/// are non-negative Euclidean distances, and column j is the distance to center
/// j (so argmin == predict, checked above). Live Probe C:
/// `m.transform(X).shape == (9, 3)`.
#[test]
fn green_req3_transform_shape_and_nonneg() {
    let x = Array2::from_shape_vec(
        (9, 2),
        vec![
            1.0, 1.0, 1.3, 0.8, 0.7, 1.2, 50.0, 1.0, 50.2, 1.3, 49.8, 0.7, 25.0, 40.0, 25.4, 39.6,
            24.6, 40.3,
        ],
    )
    .unwrap();

    let model = KMeans::<f64>::new(3).with_random_state(0).with_n_init(5);
    let fitted = model.fit(&x, &()).unwrap();
    let dists = fitted.transform(&x).unwrap();

    // sklearn: transform(X).shape == (n_samples, n_clusters) == (9, 3).
    assert_eq!(
        dists.dim(),
        (9, 3),
        "transform shape must be (n_samples, n_clusters)"
    );

    for &d in dists.iter() {
        assert!(
            d >= 0.0,
            "transform distances must be non-negative Euclidean distances"
        );
    }

    // Column j is the distance to cluster_centers_[j]: verify against a direct
    // recomputation from the fitted centers (independent of the impl path).
    let centers = fitted.cluster_centers();
    for i in 0..x.nrows() {
        for j in 0..centers.nrows() {
            let mut sq = 0.0;
            for f in 0..x.ncols() {
                let diff = x[[i, f]] - centers[[j, f]];
                sq += diff * diff;
            }
            let expected = sq.sqrt();
            assert!(
                (dists[[i, j]] - expected).abs() <= 1e-9,
                "transform[{i},{j}] must be ||x_i - center_j||"
            );
        }
    }
}

// ----------------------------------------------------------------------------
// (B) PINS — FAIL against current code, go green after the owning fix.
// ----------------------------------------------------------------------------

/// PIN — REQ-14 (#1045): `n_init` constructor default.
///
/// sklearn `KMeans(n_init="auto")` resolves to `_n_init = 1` for the default
/// `init="k-means++"` (`sklearn/cluster/_kmeans.py:886-888`:
/// `if self.n_init == "auto": if ... self.init == "k-means++": self._n_init = 1`;
/// docstring `:359-361`). Live oracle:
/// `KMeans().fit(X)._n_init  ->  1`.
///
/// ferrolearn has NO `init` param (always greedy k-means++), so the
/// sklearn-matching default is 1. `fn new` currently defaults `n_init = 10`.
///
/// FAILS today (10 != 1); goes green after `fn new` sets `n_init = 1`.
/// Tracking: #1045
#[test]
fn pin_req14_n_init_default_is_one() {
    // sklearn _check_params_vs_input (_kmeans.py:886-888): n_init="auto" +
    // init="k-means++" => _n_init = 1. Symbolic constant from sklearn file:line.
    const SKLEARN_DEFAULT_N_INIT_KMEANS_PP: usize = 1;

    let model = KMeans::<f64>::new(3);
    assert_eq!(
        model.n_init, SKLEARN_DEFAULT_N_INIT_KMEANS_PP,
        "KMeans::new default n_init must match sklearn's effective 1 for init=k-means++ \
         (_kmeans.py:886-888)"
    );
}

/// PIN — REQ-6 (#1037): `fit(X).predict(X) == labels_`.
///
/// sklearn re-runs the E-step after the Lloyd loop (when not strict-converged)
/// so the stored `labels_` match the final `cluster_centers_`
/// (`sklearn/cluster/_kmeans.py:605-625`), guaranteeing
/// `np.array_equal(m.predict(X), m.labels_) == True` for ANY seed / max_iter.
/// Verified universal in the live oracle on this exact fixture (all seeds /
/// max_iter -> True).
///
/// ferrolearn `fn fit` assigns `labels` against the PRE-swap centers, then
/// `std::mem::swap`s in the recomputed `new_centers` and stores THOSE as
/// `cluster_centers_` while keeping the stale `labels`. `predict` re-assigns to
/// the (post-swap) `cluster_centers_`, so it can disagree with `labels_`.
///
/// Fixture (deterministic, RNG-free at the contract level — the invariant must
/// hold for any seed): 12 overlapping line-pair points, k=3, random_state=0,
/// n_init=1, max_iter=1. With current code:
///   labels_  = [1,1,1,2,2,2,0,0,0,0,0,0]
///   predict  = [1,1,1,2,2,2,2,0,0,0,0,0]   (row 6 disagrees)
/// (ferrolearn observed output — NOT an oracle value; the ASSERTED contract is
/// the sklearn-guaranteed `predict == labels_` invariant.)
///
/// FAILS today; goes green after the post-loop E-step re-run lands (`labels_`
/// reassigned to the final centers before storage), which makes `predict ==
/// labels_` hold by construction.
/// Tracking: #1037
#[test]
fn pin_req6_predict_equals_labels() {
    let x = Array2::from_shape_vec(
        (12, 2),
        vec![
            0.0, 0.0, 1.0, 0.5, 2.0, 0.0, 3.0, 1.0, 4.0, 0.0, 5.0, 1.0, 6.0, 0.0, 7.0, 1.0, 8.0,
            0.0, 9.0, 1.0, 10.0, 0.0, 11.0, 1.0,
        ],
    )
    .unwrap();

    let model = KMeans::<f64>::new(3)
        .with_random_state(0)
        .with_n_init(1)
        .with_max_iter(1);
    let fitted = model.fit(&x, &()).unwrap();

    let pred = fitted.predict(&x).unwrap();
    let labels = fitted.labels();

    // sklearn invariant (_kmeans.py:605-625): predict(X) == labels_.
    assert_eq!(
        pred.to_vec(),
        labels.to_vec(),
        "fit(X).predict(X) must equal labels_ (sklearn post-loop E-step, _kmeans.py:605-625)"
    );
}
