//! Divergence + green-guard tests for `ferrolearn-cluster::MiniBatchKMeans`
//! against the live scikit-learn 1.5.2 oracle (`sklearn/cluster/_kmeans.py`,
//! `class MiniBatchKMeans(_BaseKMeans)` :1687).
//!
//! Two groups:
//!
//! **(A) GREEN-GUARDS** — PASS against current code. They pin the SHIPPED
//! Rust-level contracts (design doc `.design/cluster/mini_batch_kmeans.md`
//! REQ-1/2/3), using expected values computed by the live sklearn 1.5.2 oracle
//! (R-CHAR-3) and canonicalized to ignore the label-integer permutation (REQ-9
//! diverges). Fresh separable fixtures NOT in-tree (`make_blobs` is the only
//! in-tree fixture — these are distinct).
//!
//! **(B) PIN** — FAILS against current code, goes green after the owning fix:
//!   - REQ-5 (#1047): `MiniBatchKMeans::new` defaults `n_init = 1`. sklearn
//!     `MiniBatchKMeans(n_init="auto")` resolves to 1 for the default
//!     `init="k-means++"` (`_kmeans.py:886-888`:
//!     `n_init=="auto"` & init str `"k-means++"` -> `self._n_init = 1`;
//!     docstring `:1778-1780`). The `default_n_init=3` passed by
//!     `MiniBatchKMeans._check_params_vs_input` (`:1923`) applies ONLY to
//!     `init="random"`/callable. Live oracle confirmed `_n_init == 1`.
//!     ferrolearn `fn new` currently defaults `n_init = 3` (a mis-translation;
//!     the `fn new` doc comment claiming n_init=3 matches sklearn is wrong and
//!     must be corrected, R-HONEST-4).
//!
//! All sklearn expected values were produced from the installed sklearn 1.5.2
//! oracle (run from /tmp), never literal-copied from the ferrolearn side
//! (R-CHAR-3).

use ferrolearn_cluster::MiniBatchKMeans;
use ferrolearn_core::{Fit, Predict, Transform};
use ndarray::Array2;

/// Canonicalize a label vector to a permutation-invariant partition signature:
/// each distinct label is renamed to the index of its first occurrence. Two
/// label vectors describe the same PARTITION iff their canonical forms are
/// equal. This compares ferrolearn's partition against sklearn's without
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
/// (0,0) and (10,10), 4 points each, batch_size = n_samples (all points used).
/// Live oracle (sklearn 1.5.2, run from /tmp):
/// ```text
/// python3 -c "import numpy as np; from sklearn.cluster import MiniBatchKMeans; \
///   X=np.array([[0.,0.],[0.1,0.1],[-0.1,0.1],[0.05,-0.05],\
///               [10.,10.],[10.1,10.1],[9.9,10.1],[10.05,9.95]]); \
///   print(MiniBatchKMeans(n_clusters=2,random_state=0,n_init=5,batch_size=8)\
///         .fit(X).labels_.tolist())"
/// # [1, 1, 1, 1, 0, 0, 0, 0]   -> canonical [0,0,0,0,1,1,1,1]
/// ```
/// Divergence basis: `sklearn/cluster/_kmeans.py:1687` (class MiniBatchKMeans);
/// partition contract REQ-1.
#[test]
fn green_req1_partition_two_blobs() {
    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            0.0, 0.0, 0.1, 0.1, -0.1, 0.1, 0.05, -0.05, // blob near (0,0)
            10.0, 10.0, 10.1, 10.1, 9.9, 10.1, 10.05, 9.95, // blob near (10,10)
        ],
    )
    .unwrap();

    let model = MiniBatchKMeans::<f64>::new(2)
        .with_random_state(0)
        .with_n_init(5)
        .with_batch_size(8);
    let fitted = model.fit(&x, &()).unwrap();

    // sklearn labels_ canonicalize to [0,0,0,0,1,1,1,1].
    let expected = vec![0usize, 0, 0, 0, 1, 1, 1, 1];
    assert_eq!(canonicalize(&fitted.labels().to_vec()), expected);
}

/// REQ-1 green-guard (3-blob partition up-to-permutation).
///
/// Fresh separable fixture: three tight blobs near (0,0), (10,10), (5,-8).
/// Live oracle:
/// ```text
/// python3 -c "import numpy as np; from sklearn.cluster import MiniBatchKMeans; \
///   X=np.array([[0.,0.],[0.1,0.1],[-0.1,0.1],[10.,10.],[10.1,10.1],[9.9,10.1],\
///               [5.,-8.],[5.1,-7.9],[4.9,-8.1]]); \
///   print(MiniBatchKMeans(n_clusters=3,random_state=0,n_init=5,batch_size=9)\
///         .fit(X).labels_.tolist())"
/// # [0, 0, 0, 1, 1, 1, 2, 2, 2]   -> canonical [0,0,0,1,1,1,2,2,2]
/// ```
/// Divergence basis: `sklearn/cluster/_kmeans.py:1687`; partition REQ-1.
#[test]
fn green_req1_partition_three_blobs() {
    let x = Array2::from_shape_vec(
        (9, 2),
        vec![
            0.0, 0.0, 0.1, 0.1, -0.1, 0.1, // blob near (0,0)
            10.0, 10.0, 10.1, 10.1, 9.9, 10.1, // blob near (10,10)
            5.0, -8.0, 5.1, -7.9, 4.9, -8.1, // blob near (5,-8)
        ],
    )
    .unwrap();

    let model = MiniBatchKMeans::<f64>::new(3)
        .with_random_state(0)
        .with_n_init(5)
        .with_batch_size(9);
    let fitted = model.fit(&x, &()).unwrap();

    // sklearn labels_ canonicalize to [0,0,0,1,1,1,2,2,2].
    let expected = vec![0usize, 0, 0, 1, 1, 1, 2, 2, 2];
    assert_eq!(canonicalize(&fitted.labels().to_vec()), expected);
}

/// REQ-2 green-guard (`predict` nearest-center contract + `predict==labels_`).
///
/// Mirrors `_BaseKMeans.predict` (`_kmeans.py:1072`). Live oracle confirms both
/// invariants hold for sklearn (independent of RNG parity):
/// ```text
/// python3 -c "import numpy as np; from sklearn.cluster import MiniBatchKMeans; \
///   X=...(3-blob)...; \
///   m=MiniBatchKMeans(n_clusters=3,random_state=0,n_init=5,batch_size=9).fit(X); \
///   print(np.array_equal(m.predict(X), m.labels_))"   # True
/// ```
/// ferrolearn `fn fit` stores `labels_` via `assign_clusters_mb` against the
/// final `cluster_centers_`, so `predict(X) == labels_` holds by construction.
#[test]
fn green_req2_predict_equals_labels() {
    let x = Array2::from_shape_vec(
        (9, 2),
        vec![
            0.0, 0.0, 0.1, 0.1, -0.1, 0.1, 10.0, 10.0, 10.1, 10.1, 9.9, 10.1, 5.0, -8.0, 5.1, -7.9,
            4.9, -8.1,
        ],
    )
    .unwrap();

    let model = MiniBatchKMeans::<f64>::new(3)
        .with_random_state(0)
        .with_n_init(5)
        .with_batch_size(9);
    let fitted = model.fit(&x, &()).unwrap();

    // sklearn: predict(X) == labels_ (True).
    assert_eq!(
        fitted.predict(&x).unwrap().to_vec(),
        fitted.labels().to_vec()
    );
}

/// REQ-2/REQ-3 green-guard (`transform(X).argmin(1) == predict(X)`).
///
/// Mirrors `_BaseKMeans.transform`/`predict` (`_kmeans.py:1130`,`:1072`). Live
/// oracle:
/// ```text
/// python3 -c "...; m=MiniBatchKMeans(n_clusters=3,random_state=0,n_init=5,\
///   batch_size=9).fit(X); \
///   print(np.array_equal(m.transform(X).argmin(1), m.predict(X)))"   # True
/// ```
/// transform returns per-center euclidean distance; its row-wise argmin is the
/// nearest-center label = predict.
#[test]
fn green_req3_transform_argmin_equals_predict() {
    let x = Array2::from_shape_vec(
        (9, 2),
        vec![
            0.0, 0.0, 0.1, 0.1, -0.1, 0.1, 10.0, 10.0, 10.1, 10.1, 9.9, 10.1, 5.0, -8.0, 5.1, -7.9,
            4.9, -8.1,
        ],
    )
    .unwrap();

    let model = MiniBatchKMeans::<f64>::new(3)
        .with_random_state(0)
        .with_n_init(5)
        .with_batch_size(9);
    let fitted = model.fit(&x, &()).unwrap();

    let dists = fitted.transform(&x).unwrap();
    let preds = fitted.predict(&x).unwrap();

    let (n, k) = dists.dim();
    let argmin: Vec<usize> = (0..n)
        .map(|i| {
            let mut best = 0usize;
            let mut best_d = dists[[i, 0]];
            for c in 1..k {
                if dists[[i, c]] < best_d {
                    best_d = dists[[i, c]];
                    best = c;
                }
            }
            best
        })
        .collect();

    // sklearn: transform(X).argmin(1) == predict(X) (True).
    assert_eq!(argmin, preds.to_vec());
}

/// REQ-3 green-guard (`transform` shape + non-negativity).
///
/// Mirrors `_BaseKMeans.transform` (`_kmeans.py:1130`). Live oracle:
/// ```text
/// python3 -c "...; print(MiniBatchKMeans(n_clusters=3,random_state=0,n_init=5,\
///   batch_size=9).fit(X).transform(X).shape)"   # (9, 3)
/// ```
/// Euclidean distances are non-negative; shape is (n_samples, n_clusters).
#[test]
fn green_req3_transform_shape_nonneg() {
    let x = Array2::from_shape_vec(
        (9, 2),
        vec![
            0.0, 0.0, 0.1, 0.1, -0.1, 0.1, 10.0, 10.0, 10.1, 10.1, 9.9, 10.1, 5.0, -8.0, 5.1, -7.9,
            4.9, -8.1,
        ],
    )
    .unwrap();

    let model = MiniBatchKMeans::<f64>::new(3)
        .with_random_state(0)
        .with_n_init(5)
        .with_batch_size(9);
    let fitted = model.fit(&x, &()).unwrap();
    let dists = fitted.transform(&x).unwrap();

    // sklearn transform(X).shape == (9, 3).
    assert_eq!(dists.dim(), (9, 3));
    for &d in dists.iter() {
        assert!(d >= 0.0, "transform distances must be non-negative");
    }
}

// ----------------------------------------------------------------------------
// (B) PIN — REQ-5 (#1047). FAILS until `fn new` defaults n_init = 1.
// ----------------------------------------------------------------------------

/// REQ-5 PIN (#1047): `MiniBatchKMeans::new` must default `n_init = 1`.
///
/// sklearn `MiniBatchKMeans(n_init="auto")` resolves to 1 for the default
/// `init="k-means++"`:
///   `sklearn/cluster/_kmeans.py:886-888`:
///   ```python
///   if self.n_init == "auto":
///       if isinstance(self.init, str) and self.init == "k-means++":
///           self._n_init = 1
///   ```
/// The `default_n_init=3` passed by `MiniBatchKMeans._check_params_vs_input`
/// (`:1923`) applies ONLY to `init="random"`/callable (`:889-892`), NOT to the
/// default k-means++. Live oracle confirmed:
/// ```text
/// python3 -c "from sklearn.cluster import MiniBatchKMeans; import numpy as np; \
///   print(MiniBatchKMeans(n_clusters=3,random_state=0).fit(np.zeros((10,2)))._n_init)"
/// # 1
/// ```
/// ferrolearn `fn new` currently defaults `n_init = 3` (mis-translation). This
/// pin is deterministic and RNG-free. NOT a tautology: the expected value `1`
/// is the live-oracle-resolved sklearn default, NOT copied from ferrolearn
/// (ferrolearn currently yields 3).
#[test]
fn pin_req5_n_init_default_is_one() {
    // sklearn `MiniBatchKMeans()._n_init == 1` for init="k-means++"
    // (_kmeans.py:886-888; live oracle confirmed).
    assert_eq!(MiniBatchKMeans::<f64>::new(3).n_init, 1);
}
