//! Divergence pins + value-contract guards for `SpectralClustering` /
//! `FittedSpectralClustering` (`ferrolearn-cluster/src/spectral.rs`) against the
//! LIVE scikit-learn 1.5.2 oracle (`from sklearn.cluster import
//! SpectralClustering`, mirroring `sklearn/cluster/_spectral.py` +
//! `sklearn/manifold/_spectral_embedding.py`).
//!
//! Every expected value below is a LIVE `sklearn` 1.5.2 oracle value (computed
//! via `python3 -c "..."` run from `/tmp`, quoted above each block) — NEVER
//! copied from the ferrolearn side (goal.md R-CHAR-3).
//!
//! Mirrors `sklearn/cluster/_spectral.py`:
//! `SpectralClustering._parameter_constraints` (_spectral.py:606-631) —
//! `"n_clusters": [Interval(Integral, 1, None, closed="left")]` (_spectral.py:607)
//! and `"gamma": [Interval(Real, 0, None, closed="left")]` (_spectral.py:612).
//! So `gamma=0.0` is INSIDE the closed-left interval and ACCEPTED at fit
//! (RBF → all-ones affinity), `gamma<0` is a HARD reject
//! (`InvalidParameterError`), and `n_clusters` must be `>= 1`.
//!
//! Design doc: `.design/cluster/spectral.md` (commit 48ac2d16).
//!
//! ## Test taxonomy
//!
//! RED pin (FAILS now, LIVE `#[test]`, NO `#[ignore]` — goes green when the
//! generator lands the single-file fix):
//! - `divergence_spectral_gamma_zero_allowed` (RED, #930) — sklearn ACCEPTS
//!   `gamma=0.0`; ferrolearn's `gamma <= F::zero()` guard (spectral.rs:241)
//!   OVER-rejects it.
//!
//! GREEN guards (PASS now — protect the parts already correct):
//! - `green_spectral_gamma_negative_rejected` (GREEN, #930) — both reject
//!   `gamma<0`; guards that the gamma=0 fix does NOT also start accepting
//!   negatives.
//! - `green_spectral_n_clusters_zero_rejected` (GREEN) — both reject
//!   `n_clusters=0` (sklearn constraint `Interval[1, inf)`, _spectral.py:607).
//! - `green_spectral_insufficient_samples` (GREEN) — ferrolearn rejects
//!   `n_samples < n_clusters`.
//!
//! ## Documented NOT-STARTED (NO forced test — RNG/eigensolver-dependent or
//! large reimplementation / missing surface):
//!
//! core spectral_embedding label parity (#929): ferrolearn's pipeline is a
//! SIMPLIFIED variant — RBF affinity → normalized ADJACENCY `D^{-1/2}AD^{-1/2}`
//! TOP-k eigenvectors (spectral.rs `normalized_laplacian` / `top_k_eigenvectors`)
//! → ROW-L2-normalize (`row_normalize`) → ferrolearn `KMeans`. sklearn uses
//! `_spectral_embedding` (`sklearn/manifold/_spectral_embedding.py:300-469`):
//! normalized Laplacian `L = I - D^{-1/2}AD^{-1/2}` (`csgraph_laplacian(normed=True)`,
//! _spectral_embedding.py:333), SMALLEST-eigenvalue eigenvectors,
//! `embedding = embedding / dd` (_spectral_embedding.py:378/443),
//! `_deterministic_vector_sign_flip` (_spectral_embedding.py:465). The eigenvector
//! SUBSPACE coincides (top-k of `D^{-1/2}AD^{-1/2}` == bottom-k of
//! `I - D^{-1/2}AD^{-1/2}`), so well-separated data clusters the same — but the
//! embedding VALUES diverge (sklearn row magnitude `1/dd ≈ 0.158`, ferrolearn
//! exactly `1.0`), and the final labels additionally depend on a DIFFERENT KMeans
//! impl with its own init/RNG (#934). Exact label parity is therefore
//! fixture-dependent and NOT a contract: it agrees on the blobs and on
//! `make_circles(...)` at `gamma=0.1`, but DIVERGES at `gamma=10` (2/30 points
//! differ — design doc Probe 1). NOT pinned as RED (a label-equality test would be
//! flaky and the fix is a large reimplementation gated on #934). Documented only.
//!
//! affinity modes (#931): sklearn `affinity ∈ {'rbf','nearest_neighbors',
//! 'precomputed','precomputed_nearest_neighbors'} ∪ KERNEL_PARAMS`
//! (_spectral.py:613-619); ferrolearn supports RBF only. Missing surface.
//!
//! assign_labels modes (#932): sklearn `assign_labels ∈ {'kmeans','discretize',
//! 'cluster_qr'}` (_spectral.py:625, branch :755-766); ferrolearn supports kmeans
//! only. Missing surface.
//!
//! missing params + n_clusters=8 default (#933): sklearn `__init__`
//! (_spectral.py:633-666) takes 16 params with `n_clusters=8` default
//! (_spectral.py:635); `SpectralClustering<F>` has only `n_clusters` (REQUIRED, no
//! default) / `gamma` / `n_init` / `random_state`. Missing surface.
//!
//! affinity_matrix_ accessor + KMeans dependency (#934): sklearn exposes
//! `affinity_matrix_` / `n_features_in_` (_spectral.py:524-538); the RBF affinity
//! VALUE matches sklearn `rbf_kernel(X, gamma=gamma)` to full f64 precision (design
//! doc `[0,1]=0.9950124791926823`), but `fn affinity_matrix` (spectral.rs:148) is a
//! PRIVATE helper with no public accessor — NOT test-reachable without inventing
//! surface, so the affinity value guard is SKIPPED (see note below). KMeans parity
//! (k-means++ init + RNG) is a separate unit. Missing surface / separate unit.
//!
//! PyO3 + ferray (#935): `grep -rln SpectralClustering ferrolearn-python/` is
//! EMPTY — no `_RsSpectralClustering`, so `import ferrolearn` cannot reach it;
//! `spectral.rs` imports `ndarray` + `num-traits` + `NdarrayFaerBackend`, not
//! `ferray-core`. Missing binding / substrate.
//!
//! ## OPTIONAL GREEN — RBF affinity VALUE: SKIPPED (NOT test-reachable).
//! The RBF affinity `exp(-gamma*||xi-xj||^2)` value-matches sklearn
//! `rbf_kernel(X, gamma=gamma)` to ~1e-12 (design doc confirmed
//! `[0,1]=0.9950124791926823`), but it is computed by the PRIVATE `fn
//! affinity_matrix` (spectral.rs:148) with NO public accessor (#934). Reaching it
//! from this test would require inventing surface (an accessor), which the critic
//! does NOT do. The value matches but is not test-reachable — guard omitted.

use ferrolearn_cluster::SpectralClustering;
use ferrolearn_core::Fit;
use ndarray::Array2;

/// Two well-separated 2-D blobs — the `two_blobs()` fixture from the in-tree
/// `spectral.rs` tests (5 points near the origin + 5 near `(10,10)`). Used so
/// that, once the gamma=0 guard is relaxed, `fit` produces a valid result rather
/// than tripping a later numerical edge.
fn two_blobs() -> Array2<f64> {
    Array2::from_shape_vec(
        (10, 2),
        vec![
            0.0, 0.0, 0.2, 0.1, -0.1, 0.2, 0.1, -0.1, 0.0, 0.1, 10.0, 10.0, 10.2, 10.1, 9.9, 10.2,
            10.1, 9.9, 10.0, 10.1,
        ],
    )
    .unwrap()
}

// ===========================================================================
// RED #930 — gamma = 0.0 is ALLOWED at fit (over-rejection in ferrolearn).
//
// sklearn `SpectralClustering._parameter_constraints["gamma"] =
// Interval(Real, 0, None, closed="left")` (_spectral.py:612) → the interval is
// [0.0, inf), so `gamma=0.0` is INSIDE it and ACCEPTED at fit (RBF kernel
// collapses to an all-ones affinity matrix; the fit still runs and produces
// labels). Only `gamma<0` is a hard reject.
//
// ferrolearn `fn fit` (spectral.rs:241) rejects `self.gamma <= F::zero()` → it
// OVER-rejects `gamma=0.0` (returns `FerroError::InvalidParameter`).
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "import numpy as np; from sklearn.cluster import SpectralClustering
//     X=np.array([[0.,0.],[0.2,0.1],[-0.1,0.2],[0.1,-0.1],[0.,0.1],
//                 [10.,10.],[10.2,10.1],[9.9,10.2],[10.1,9.9],[10.,10.1]])
//     m=SpectralClustering(n_clusters=2,gamma=0.0,affinity='rbf',random_state=42).fit(X)
//     print('gamma=0 fit ok; labels', m.labels_.tolist())"
//   ->  gamma=0 fit ok; labels [0, 1, 1, 1, 0, 0, 0, 1, 0, 1]   (NO error)
//
// We pin the OBSERVABLE: `fit` must return `Ok` (sklearn accepts). ferrolearn
// currently returns `Err`: FAILS now. Minimally fixable in `spectral.rs` `fn fit`
// — change the line-241 guard from `gamma <= F::zero()` to `gamma < F::zero()`
// (allow zero, reject only negatives).
// ===========================================================================

/// Divergence: ferrolearn's `SpectralClustering::fit` REJECTS `gamma = 0.0`,
/// whereas `sklearn/cluster/_spectral.py:612`
/// (`"gamma": [Interval(Real, 0, None, closed="left")]`) makes `gamma = 0.0`
/// ALLOWED (inside the closed-left interval `[0.0, inf)`).
/// `SpectralClustering(n_clusters=2, gamma=0.0, random_state=42).fit(X)` SUCCEEDS
/// in sklearn (RBF → all-ones affinity, NO error); ferrolearn returns `Err`
/// because the spectral.rs:241 guard `gamma <= F::zero()` fires.
/// Tracking: #930
#[test]
fn divergence_spectral_gamma_zero_allowed() {
    let x = two_blobs();

    // gamma = 0.0 — sklearn ACCEPTS (oracle above: "gamma=0 fit ok").
    let model = SpectralClustering::<f64>::new(2)
        .with_gamma(0.0)
        .with_random_state(42);
    let result = model.fit(&x, &());

    assert!(
        result.is_ok(),
        "fit with gamma=0.0 should SUCCEED (sklearn: gamma Interval closed-left \
         at 0 → 0.0 ACCEPTED, RBF collapses to all-ones affinity, NO error), got \
         Err (#930)"
    );
}

// ===========================================================================
// GREEN #930 — gamma < 0 is REJECTED at fit (both sides reject).
//
// sklearn `gamma: Interval(Real, 0, None, closed="left")` (_spectral.py:612) →
// `gamma=-1.0` is OUTSIDE [0.0, inf) and raises `InvalidParameterError`.
// ferrolearn `fn fit` (spectral.rs:241, `gamma <= F::zero()`) also rejects a
// negative gamma → both error. This guard ensures the #930 fix (relaxing
// `<= 0` to `< 0`) does NOT regress into accepting NEGATIVE gamma.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "import numpy as np; from sklearn.cluster import SpectralClustering
//     X=np.array([[0.,0.],[0.2,0.1],[-0.1,0.2],[0.1,-0.1],[0.,0.1],
//                 [10.,10.],[10.2,10.1],[9.9,10.2],[10.1,9.9],[10.,10.1]])
//     try:
//         SpectralClustering(n_clusters=2,gamma=-1.0,random_state=42).fit(X)
//         print('no error')
//     except Exception as e:
//         print(type(e).__name__)"
//   ->  InvalidParameterError
//       (full msg: "The 'gamma' parameter of SpectralClustering must be a float
//        in the range [0.0, inf). Got -1.0 instead.")
// ===========================================================================

/// Guard: ferrolearn `with_gamma(-1.0).fit` returns `Err`, matching sklearn which
/// raises `InvalidParameterError` for `gamma=-1.0` (outside
/// `Interval(Real, 0, None, closed="left")`, _spectral.py:612). Guards that the
/// #930 fix accepts `gamma=0` WITHOUT also accepting negative gamma.
#[test]
fn green_spectral_gamma_negative_rejected() {
    let x = two_blobs();

    // gamma = -1.0 — sklearn raises InvalidParameterError (oracle above).
    let result = SpectralClustering::<f64>::new(2)
        .with_gamma(-1.0)
        .with_random_state(42)
        .fit(&x, &());

    assert!(
        result.is_err(),
        "fit with gamma=-1.0 should error (sklearn: InvalidParameterError, gamma \
         must be in [0.0, inf)), got Ok"
    );
}

// ===========================================================================
// GREEN — n_clusters = 0 is REJECTED at fit (both sides reject).
//
// sklearn `n_clusters: Interval(Integral, 1, None, closed="left")`
// (_spectral.py:607) → `n_clusters=0` is OUTSIDE [1, inf) and raises
// `InvalidParameterError`. ferrolearn `fn fit` (spectral.rs:235) rejects
// `n_clusters == 0` → both error.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "import numpy as np; from sklearn.cluster import SpectralClustering
//     X=np.array([[0.,0.],[0.2,0.1],[-0.1,0.2],[0.1,-0.1],[0.,0.1],
//                 [10.,10.],[10.2,10.1],[9.9,10.2],[10.1,9.9],[10.,10.1]])
//     try:
//         SpectralClustering(n_clusters=0,random_state=42).fit(X)
//         print('no error')
//     except Exception as e:
//         print(type(e).__name__)"
//   ->  InvalidParameterError
//       (full msg: "The 'n_clusters' parameter of SpectralClustering must be an
//        int in the range [1, inf). Got 0 instead.")
// ===========================================================================

/// Guard: ferrolearn `new(0).fit` returns `Err`, matching sklearn which raises
/// `InvalidParameterError` for `n_clusters=0` (outside
/// `Interval(Integral, 1, None, closed="left")`, _spectral.py:607).
#[test]
fn green_spectral_n_clusters_zero_rejected() {
    let x = two_blobs();

    // n_clusters = 0 — sklearn raises InvalidParameterError (oracle above).
    let result = SpectralClustering::<f64>::new(0)
        .with_random_state(42)
        .fit(&x, &());

    assert!(
        result.is_err(),
        "fit with n_clusters=0 should error (sklearn: InvalidParameterError, \
         n_clusters must be in [1, inf)), got Ok"
    );
}

// ===========================================================================
// GREEN — n_samples < n_clusters is REJECTED at fit.
//
// A spectral clustering with more requested clusters than samples cannot be
// fit: sklearn's `_spectral_embedding` / `k_means` cannot place 3 centroids on
// 1 sample (raises downstream). ferrolearn `fn fit` (spectral.rs:254) guards
// `n_samples < n_clusters` → `Err(InsufficientSamples)`.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "import numpy as np; from sklearn.cluster import SpectralClustering
//     X=np.array([[0.,0.]])
//     try:
//         SpectralClustering(n_clusters=3,random_state=42).fit(X)
//         print('no error')
//     except Exception as e:
//         print(type(e).__name__)"
//   ->  ValueError   (1 sample cannot support 3 clusters; sklearn errors)
// ===========================================================================

/// Guard: ferrolearn `new(3).fit(X_1row)` returns `Err` (n_samples=1 <
/// n_clusters=3), matching sklearn which errors when there are fewer samples than
/// requested clusters.
#[test]
fn green_spectral_insufficient_samples() {
    // 1 sample, 3 clusters requested — sklearn errors (oracle above).
    let x = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
    let result = SpectralClustering::<f64>::new(3)
        .with_random_state(42)
        .fit(&x, &());

    assert!(
        result.is_err(),
        "fit with n_samples=1 < n_clusters=3 should error (cannot place 3 \
         clusters on 1 sample), got Ok"
    );
}
