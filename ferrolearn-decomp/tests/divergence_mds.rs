//! Divergence / green-guard audit for `ferrolearn_decomp::mds`
//! (`MDS` / `FittedMDS`, classical metric MDS / PCoA) against scikit-learn
//! 1.5.2 `sklearn/manifold/_mds.py`.
//!
//! CRITICAL CARVE-OUT (REQ-1, #1451-A): ferrolearn implements CLASSICAL
//! (metric) MDS / PCoA via closed-form eigendecomposition of the double-centred
//! Gram matrix (`classical_mds`, `mds.rs:195`). sklearn `MDS`
//! (`_mds.py:395`) is SMACOF: iterative stress majorization
//! (`_smacof_single`, `_mds.py:22-167`) from `n_init=4` numpy-`RandomState`
//! random inits (`X = random_state.uniform(...)`, `_mds.py:113`), keeping the
//! lowest-stress run (`_mds.py:363-365`). EXACT coordinate parity is therefore
//! structurally IMPOSSIBLE (different algorithm + RNG + arbitrary
//! rotation/reflection). No coordinate-parity test exists here by design
//! (R-DEFER-3).
//!
//! The verifiable COMMON property both algorithms target is DISTANCE
//! PRESERVATION (REQ-2): the embedding's pairwise Euclidean distances
//! reconstruct the INPUT pairwise distance matrix. The oracle for that is the
//! INPUT distance matrix itself (`euclidean_distances`, computed live from
//! sklearn 1.5.2 in `/tmp`), NOT sklearn's SMACOF coordinates (R-CHAR-3).
//!
//! These are GREEN-GUARDS: they MUST PASS where ferrolearn holds. A FAIL marks
//! a genuine divergence in the SHIPPED classical-MDS scope.

use ferrolearn_core::traits::Fit;
use ferrolearn_decomp::{Dissimilarity, MDS};
use ndarray::{Array2, array};

/// Compute the pairwise Euclidean distance matrix of an embedding, in-test.
fn pairwise_euclidean(emb: &Array2<f64>) -> Array2<f64> {
    let n = emb.nrows();
    let mut d = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in (i + 1)..n {
            let mut sq = 0.0;
            for k in 0..emb.ncols() {
                let diff = emb[[i, k]] - emb[[j, k]];
                sq += diff * diff;
            }
            let dist = sq.sqrt();
            d[[i, j]] = dist;
            d[[j, i]] = dist;
        }
    }
    d
}

/// Assert two distance matrices are elementwise equal within `tol`.
fn assert_dist_eq(actual: &Array2<f64>, expected: &Array2<f64>, tol: f64) {
    assert_eq!(
        actual.dim(),
        expected.dim(),
        "distance-matrix shape mismatch"
    );
    let n = expected.nrows();
    for i in 0..n {
        for j in 0..n {
            let a = actual[[i, j]];
            let e = expected[[i, j]];
            assert!(
                (a - e).abs() <= tol,
                "distance [{i},{j}] = {a}, expected {e} (|diff| = {}, tol {tol})",
                (a - e).abs()
            );
        }
    }
}

// ---------------------------------------------------------------------------
// (a) DISTANCE-PRESERVATION at FULL RANK — headline SHIPPED claim (REQ-2).
// ---------------------------------------------------------------------------

/// Green-guard (REQ-2, AC-2): the classical-MDS embedding of the 3-4-5
/// rectangle `[[0,0],[3,0],[0,4],[3,4]]` at `n_components=2` (full rank for 2D
/// data) reconstructs the INPUT pairwise distance matrix EXACTLY.
///
/// Oracle (sklearn 1.5.2, `/tmp`):
///   `euclidean_distances([[0,0],[3,0],[0,4],[3,4]])`
///   = `[[0,3,4,5],[3,0,5,4],[4,5,0,3],[5,4,3,0]]`.
/// This is the INPUT distance matrix — the ground truth any MDS must preserve
/// (R-CHAR-3) — NOT sklearn's SMACOF coordinates.
/// Mirrors the stress objective sklearn's `_smacof_single` (`_mds.py:147`)
/// minimizes, achieved here in closed form (`classical_mds`, `mds.rs:195`).
#[test]
#[allow(
    clippy::assertions_on_constants,
    reason = "assert!(false, ...) reports the unexpected fit Err with diagnostics"
)]
fn green_distance_preservation_full_rank_2d() {
    let x = array![[0.0, 0.0], [3.0, 0.0], [0.0, 4.0], [3.0, 4.0]];
    let expected_input_dist: Array2<f64> = array![
        [0.0, 3.0, 4.0, 5.0],
        [3.0, 0.0, 5.0, 4.0],
        [4.0, 5.0, 0.0, 3.0],
        [5.0, 4.0, 3.0, 0.0],
    ];

    let fitted = match MDS::new(2).fit(&x, &()) {
        Ok(f) => f,
        Err(e) => {
            assert!(
                false,
                "MDS::new(2).fit failed on full-rank 2D fixture: {e:?}"
            );
            return;
        }
    };
    let emb_dist = pairwise_euclidean(fitted.embedding());
    assert_dist_eq(&emb_dist, &expected_input_dist, 1e-9);
}

// ---------------------------------------------------------------------------
// (b) DISTANCE-PRESERVATION on a 3D->3D Euclidean fixture (full rank).
// ---------------------------------------------------------------------------

/// Green-guard (REQ-2): a full-rank 3D dataset embedded at `n_components=3`
/// reconstructs the INPUT distance matrix exactly.
///
/// Oracle (sklearn 1.5.2, `/tmp`):
///   `euclidean_distances([[0,0,0],[2,0,0],[0,3,0],[0,0,4]])` =
///   `[[0, 2, 3, 4],
///     [2, 0, sqrt(13)=3.605551275464, sqrt(20)=4.472135955],
///     [3, 3.605551275464, 0, 5],
///     [4, 4.472135955, 5, 0]]`.
#[test]
#[allow(
    clippy::assertions_on_constants,
    reason = "assert!(false, ...) reports the unexpected fit Err with diagnostics"
)]
fn green_distance_preservation_full_rank_3d() {
    let x = array![
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, 0.0, 4.0],
    ];
    let s13 = 13.0_f64.sqrt(); // 3.605551275463989
    let s20 = 20.0_f64.sqrt(); // 4.47213595499958
    let expected_input_dist: Array2<f64> = array![
        [0.0, 2.0, 3.0, 4.0],
        [2.0, 0.0, s13, s20],
        [3.0, s13, 0.0, 5.0],
        [4.0, s20, 5.0, 0.0],
    ];

    let fitted = match MDS::new(3).fit(&x, &()) {
        Ok(f) => f,
        Err(e) => {
            assert!(
                false,
                "MDS::new(3).fit failed on full-rank 3D fixture: {e:?}"
            );
            return;
        }
    };
    let emb_dist = pairwise_euclidean(fitted.embedding());
    assert_dist_eq(&emb_dist, &expected_input_dist, 1e-9);
}

// ---------------------------------------------------------------------------
// (c) REDUCED RANK best-approximation (projection — do NOT assert exact dists).
// ---------------------------------------------------------------------------

/// Green-guard (REQ-2 best-low-rank / REQ-3 shape): a near-planar 3D dataset
/// embedded at `n_components=2` yields a shape-`(4,2)` embedding whose stress
/// is finite and `>= 0`. This is a projection (rank reduction), so distances
/// are only APPROXIMATED — assert structure + finiteness, not exact distances.
#[test]
#[allow(
    clippy::assertions_on_constants,
    reason = "assert!(false, ...) reports the unexpected fit Err with diagnostics"
)]
fn green_reduced_rank_best_approximation() {
    // Near-planar (tiny z-perturbation) — best rank-2 approx is close but lossy.
    let x = array![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.01],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.01],
    ];
    let fitted = match MDS::new(2).fit(&x, &()) {
        Ok(f) => f,
        Err(e) => {
            assert!(
                false,
                "MDS::new(2).fit failed on near-planar 3D fixture: {e:?}"
            );
            return;
        }
    };
    assert_eq!(fitted.embedding().dim(), (4, 2), "embedding must be (4,2)");
    assert!(
        fitted.embedding().iter().all(|v| v.is_finite()),
        "embedding must be finite"
    );
    let s = fitted.stress();
    assert!(s.is_finite(), "stress must be finite, got {s}");
    assert!(s >= 0.0, "stress must be >= 0, got {s}");
}

// ---------------------------------------------------------------------------
// (d) STRESS ~ 0 on the full-rank exact fixture (REQ-4).
// ---------------------------------------------------------------------------

/// Green-guard (REQ-4, AC-4): a perfect (full-rank) embedding has ~zero Kruskal
/// stress-1. On fixture (a), `stress()` must be `<= 1e-6` — a perfect
/// distance reconstruction has zero stress by definition
/// (`kruskal_stress`, `mds.rs:151`).
#[test]
#[allow(
    clippy::assertions_on_constants,
    reason = "assert!(false, ...) reports the unexpected fit Err with diagnostics"
)]
fn green_stress_zero_on_perfect_embedding() {
    let x = array![[0.0, 0.0], [3.0, 0.0], [0.0, 4.0], [3.0, 4.0]];
    let fitted = match MDS::new(2).fit(&x, &()) {
        Ok(f) => f,
        Err(e) => {
            assert!(false, "MDS::new(2).fit failed: {e:?}");
            return;
        }
    };
    let s = fitted.stress();
    assert!(
        s.abs() <= 1e-6,
        "Kruskal stress-1 on perfect full-rank embedding must be ~0, got {s}"
    );
}

// ---------------------------------------------------------------------------
// (e) DETERMINISM — classical MDS has no RNG (REQ-3).
// ---------------------------------------------------------------------------

/// Green-guard (REQ-3, AC-3): classical MDS is deterministic (no RNG, unlike
/// sklearn's RNG-seeded SMACOF). Two independent fits on the same input produce
/// bitwise-identical embeddings.
#[test]
#[allow(
    clippy::assertions_on_constants,
    reason = "assert!(false, ...) reports the unexpected fit Err with diagnostics"
)]
fn green_determinism_identical_runs() {
    let x = array![[0.0, 0.0], [3.0, 0.0], [0.0, 4.0], [3.0, 4.0]];
    let f1 = match MDS::new(2).fit(&x, &()) {
        Ok(f) => f,
        Err(e) => {
            assert!(false, "first fit failed: {e:?}");
            return;
        }
    };
    let f2 = match MDS::new(2).fit(&x, &()) {
        Ok(f) => f,
        Err(e) => {
            assert!(false, "second fit failed: {e:?}");
            return;
        }
    };
    let e1 = f1.embedding();
    let e2 = f2.embedding();
    assert_eq!(e1.dim(), e2.dim(), "embedding shapes must match");
    for i in 0..e1.nrows() {
        for j in 0..e1.ncols() {
            assert_eq!(
                e1[[i, j]],
                e2[[i, j]],
                "embedding [{i},{j}] differs across runs (non-deterministic)"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// (f) PRECOMPUTED — fit on a valid Euclidean distance matrix at full rank.
// ---------------------------------------------------------------------------

/// Green-guard (REQ-9 precomputed path, REQ-2): `Dissimilarity::Precomputed`
/// fed a VALID Euclidean distance matrix `D` (the 3-4-5 rectangle's distances)
/// at full rank reconstructs `D` exactly.
///
/// Oracle (sklearn 1.5.2, `/tmp`): `D = euclidean_distances([[0,0],[3,0],
/// [0,4],[3,4]]) = [[0,3,4,5],[3,0,5,4],[4,5,0,3],[5,4,3,0]]` (R-CHAR-3 —
/// the precomputed input distances ARE the ground truth).
#[test]
#[allow(
    clippy::assertions_on_constants,
    reason = "assert!(false, ...) reports the unexpected fit Err with diagnostics"
)]
fn green_precomputed_reconstructs_distances() {
    let d: Array2<f64> = array![
        [0.0, 3.0, 4.0, 5.0],
        [3.0, 0.0, 5.0, 4.0],
        [4.0, 5.0, 0.0, 3.0],
        [5.0, 4.0, 3.0, 0.0],
    ];
    let fitted = match MDS::new(2)
        .with_dissimilarity(Dissimilarity::Precomputed)
        .fit(&d, &())
    {
        Ok(f) => f,
        Err(e) => {
            assert!(false, "precomputed fit failed: {e:?}");
            return;
        }
    };
    assert_eq!(fitted.embedding().dim(), (4, 2));
    let emb_dist = pairwise_euclidean(fitted.embedding());
    assert_dist_eq(&emb_dist, &d, 1e-9);
}

// ---------------------------------------------------------------------------
// (g) ERROR CONTRACTS (REQ-5).
// ---------------------------------------------------------------------------

/// Green-guard (REQ-5, AC-5): `n_components == 0` -> `fit` returns `Err`.
#[test]
fn green_error_n_components_zero() {
    let x = array![[0.0, 0.0], [3.0, 0.0], [0.0, 4.0], [3.0, 4.0]];
    assert!(
        MDS::new(0).fit(&x, &()).is_err(),
        "n_components=0 must error"
    );
}

/// Green-guard (REQ-5, AC-5): `n_components > n_samples` -> `fit` returns
/// `Err` (ferrolearn's scoped guard; note sklearn has no such upper bound —
/// `_mds.py:535` admits `Interval(Integral, 1, None)` — so ferrolearn is
/// stricter, FLAGGED in the design doc REQ-5, not pinned as a divergence here).
#[test]
fn green_error_n_components_exceeds_n_samples() {
    let x = array![[0.0, 0.0], [3.0, 0.0], [0.0, 4.0], [3.0, 4.0]]; // 4 samples
    assert!(
        MDS::new(10).fit(&x, &()).is_err(),
        "n_components(10) > n_samples(4) must error"
    );
}
