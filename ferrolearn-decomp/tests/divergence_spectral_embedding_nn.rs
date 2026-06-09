//! Divergence pin: `SpectralEmbedding` with `affinity='nearest_neighbors'`
//! against scikit-learn 1.5.2 `sklearn.manifold.SpectralEmbedding`
//! (`/home/doll/scikit-learn/sklearn/manifold/_spectral_embedding.py`).
//!
//! sklearn builds the nearest-neighbors affinity as
//! `affinity_matrix_ = kneighbors_graph(X, n_neighbors_, include_self=True)`
//! then symmetrizes `affinity_matrix_ = 0.5 * (A + A.T)`
//! (`_spectral_embedding.py:704-710`). `include_self=True` counts each point
//! as one of its own `n_neighbors`, and the symmetrization AVERAGES (weight
//! 0.5 for one-directional edges, 1.0 for mutual edges).
//!
//! ferrolearn's `Affinity::NearestNeighbors` instead:
//!   - EXCLUDES self when selecting the `k` nearest (so each point keeps `k`
//!     genuine neighbors rather than `k-1`),
//!   - symmetrizes by UNION with weight 1.0 (not 0.5 averaging),
//!   - sets the diagonal to 0 (irrelevant — `csgraph_laplacian` zeroes it).
//! (`ferrolearn-decomp/src/spectral_embedding.rs:202-232`).
//!
//! These different affinity graphs yield a DIFFERENT normalized Laplacian and
//! hence a DIFFERENT embedding. The divergence is NOT a free sign: it survives
//! any per-column sign flip (min max-abs difference ~0.35, far above 1e-6).
//!
//! Tracking: #2409.

use ferrolearn_core::traits::Fit;
use ferrolearn_decomp::{Affinity, SpectralEmbedding};
use ndarray::{Array2, array};

/// Connected 5-point fixture (same as the RBF green-parity fixture).
fn line5() -> Array2<f64> {
    array![[0.0, 0.0], [1.2, 0.3], [2.0, 1.1], [3.5, 0.2], [4.1, 2.0]]
}

/// Expected `embedding_` from the LIVE sklearn 1.5.2 oracle:
/// `SpectralEmbedding(n_components=2, affinity='nearest_neighbors',
///  n_neighbors=3, random_state=42).fit_transform(line5())`.
/// Column 0, then column 1.
const SK_COL0: [f64; 5] = [
    -0.408_248_290_463_862_85,
    -0.353_553_390_593_273_9,
    0.0,
    0.353_553_390_593_273_7,
    0.408_248_290_463_863,
];
const SK_COL1: [f64; 5] = [
    0.408_248_290_463_862_96,
    -7.850_462_293_418_875e-17,
    -0.408_248_290_463_863,
    -3.532_708_032_038_494e-16,
    0.408_248_290_463_863_1,
];

/// Per-column best sign-aligned max-abs difference. Returns the divergence
/// magnitude that survives an optimal per-column sign flip, so a pure free-sign
/// difference would report ~0 here. The NN-graph divergence does NOT.
fn signed_col_maxdiff(ferro: &Array2<f64>, sk: &[f64; 5], col: usize) -> f64 {
    let pos = (0..5)
        .map(|i| (ferro[[i, col]] - sk[i]).abs())
        .fold(0.0_f64, f64::max);
    let neg = (0..5)
        .map(|i| (-ferro[[i, col]] - sk[i]).abs())
        .fold(0.0_f64, f64::max);
    pos.min(neg)
}

/// Divergence: ferrolearn's `nearest_neighbors` affinity diverges from
/// `sklearn/manifold/_spectral_embedding.py:704-710`
/// (`kneighbors_graph(..., include_self=True)` then `0.5*(A+A.T)`).
/// sklearn col0 ≈ [-0.408, -0.354, 0, 0.354, 0.408]; ferrolearn produces a
/// structurally different embedding (e.g. col0 ≈ [0.408, 0, 0, 0, -0.408]).
/// The mismatch survives any per-column sign flip.
#[test]
#[ignore = "divergence: nearest_neighbors affinity (include_self + 0.5 symmetrization) mismatch; tracking #2409"]
fn divergence_spectral_embedding_nearest_neighbors_affinity() {
    let x = line5();
    let se = SpectralEmbedding::new(2).with_affinity(Affinity::NearestNeighbors { n_neighbors: 3 });
    let fitted = se.fit(&x, &()).expect("fit must succeed");
    let emb = fitted.embedding();

    let d0 = signed_col_maxdiff(emb, &SK_COL0, 0);
    let d1 = signed_col_maxdiff(emb, &SK_COL1, 1);

    // sklearn parity (R-DEV-1, ~1e-6). FAILS today: ferrolearn's NN affinity
    // graph differs from sklearn's, so the embedding differs by ~0.35.
    assert!(
        d0 < 1e-6 && d1 < 1e-6,
        "nearest_neighbors embedding diverges from sklearn: \
         col0 sign-aligned maxdiff={d0}, col1 sign-aligned maxdiff={d1} (want < 1e-6)"
    );
}
