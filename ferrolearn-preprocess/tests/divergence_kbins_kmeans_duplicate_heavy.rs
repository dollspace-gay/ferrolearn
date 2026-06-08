//! Divergence pin: `KBinsDiscretizer(strategy="kmeans")` on DUPLICATE-HEAVY data
//! where `n_bins > n_distinct(column)`. The #2321/#2322 `kmeans_1d` rewrite
//! (mean-centering + empty-cluster relocation + heaviest-cluster fallback)
//! converges to a DIFFERENT, DETERMINISTIC partition than scikit-learn 1.5.2 on
//! this regime — producing a different `n_bins_`/`bin_edges_`.
//!
//! This is DISTINCT from the honestly-pinned `divergence_km3_blas_gemm_local_optimum`
//! in `divergence_kbins_discretizer.rs`, which covers ALL-DISTINCT well-spread data where both ferrolearn and
//! sklearn reach valid-but-different Lloyd FIXED POINTS via BLAS-gemm float tie-break
//! ordering. The case below is NOT a float tie-break:
//!   - sklearn's result is RNG-INVARIANT and bit-reproducible (verified over 20 fits);
//!   - the divergence is in the `_relocate_empty_clusters_dense` /
//!     `_average_centers` heaviest-cluster fallback when `n_bins > n_distinct`
//!     (`sklearn/cluster/_k_means_common.pyx:170-214, 277-298`);
//!   - sklearn produces DUPLICATE centers (empties dumped onto the heaviest
//!     cluster) that collapse via small-bin removal to FEWER bins; ferrolearn's
//!     relocation moves "farthest" points to distinct values, keeping MORE bins.
//!
//! A 400-case live-oracle fuzz shows this regime matches sklearn only ~58.6%
//! (`n_bins>n_distinct`), vs ~96.2% for the all-distinct regime — so this is a
//! systematic, characterizable divergence, NOT noise.
//!
//! All expected values from the LIVE sklearn 1.5.2 oracle (/tmp, warnings off,
//! subsample=None). R-CHAR-3: never copied from ferrolearn.
//!
//! sklearn cite (`KBinsDiscretizer.fit`):
//!   `sklearn/preprocessing/_discretization.py:285-300`
//!     km = KMeans(n_clusters=n_bins, init=uniform-midpoints, n_init=1)
//!     centers = km.fit(col).cluster_centers_; centers.sort()
//!     bin_edges = r_[col_min, midpoints(centers), col_max]
//!   `:302-312` small-bin removal (ediff1d <= 1e-8 collapse).
//!
//! Tracking: #2323

use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_preprocess::kbins_discretizer::{BinEncoding, BinStrategy, KBinsDiscretizer};
use ndarray::{Array2, array};

/// Divergence: kmeans `bin_edges_`/`n_bins_` diverges from
/// `sklearn/preprocessing/_discretization.py:285-312` for
/// X = [[0],[0],[2],[1],[1],[2]], n_bins=5 (n_distinct=3, so n_bins > n_distinct).
///
/// LIVE ORACLE (sklearn 1.5.2, RNG-INVARIANT over 20 fits, subsample=None):
///   raw KMeans centers sorted = [0, 0, 0, 1, 2]  (two empties dumped on heaviest)
///   -> bin_edges_ = [0.0, 0.5, 1.5, 2.0]   (n_bins_ = 3, after small-bin removal)
///   -> transform   = [0, 0, 2, 1, 1, 2]
/// ferrolearn returns bin_edges_ = [0.0, 0.5, 1.0, 1.5, 2.0] (n_bins_ = 4) and so
///   transform diverges. The extra interior edge at 1.0 comes from ferrolearn's
///   relocation producing a distinct center where sklearn duplicates the heaviest.
///
/// This is a DETERMINISTIC structural divergence (different bin count), NOT the
/// BLAS-gemm float tie-break local-optimum split (`divergence_km3_blas_gemm_local_optimum`).
#[allow(
    clippy::assertions_on_constants,
    reason = "assert!(false, ...) fails the test from a Result Err arm"
)]
#[test]
#[ignore = "divergence: kmeans n_bins>n_distinct relocation/heaviest-fallback gives wrong n_bins_/edges (deterministic, NOT BLAS-order); tracking #2323"]
fn divergence_kmeans_dup_heavy_more_bins_than_distinct() {
    // sklearn oracle (RNG-invariant): 4 edges => n_bins_ = 3.
    const SK_EDGES: [f64; 4] = [0.0, 0.5, 1.5, 2.0];
    const SK_TRANSFORM: [f64; 6] = [0.0, 0.0, 2.0, 1.0, 1.0, 2.0];

    let disc = KBinsDiscretizer::<f64>::new(5, BinEncoding::Ordinal, BinStrategy::KMeans);
    let x: Array2<f64> = array![[0.0], [0.0], [2.0], [1.0], [1.0], [2.0]];
    let fitted = match disc.fit(&x, &()) {
        Ok(f) => f,
        Err(e) => {
            assert!(false, "fit failed: {e:?}");
            return;
        }
    };

    let edges = &fitted.bin_edges()[0];
    assert_eq!(
        edges.len(),
        SK_EDGES.len(),
        "dup-heavy: sklearn keeps {} edges (n_bins_=3) {SK_EDGES:?}; ferrolearn has {} {edges:?}",
        SK_EDGES.len(),
        edges.len()
    );
    for (e, &s) in edges.iter().zip(SK_EDGES.iter()) {
        assert!(
            (*e - s).abs() <= 1e-9,
            "dup-heavy: kmeans edge {e} != sklearn {s} (sklearn {SK_EDGES:?}, ferrolearn {edges:?})"
        );
    }

    let out = match fitted.transform(&x) {
        Ok(o) => o,
        Err(e) => {
            assert!(false, "transform failed: {e:?}");
            return;
        }
    };
    let got: Vec<f64> = out.iter().copied().collect();
    assert_eq!(
        got, SK_TRANSFORM,
        "dup-heavy: sklearn transform = {SK_TRANSFORM:?}; ferrolearn gave {got:?}"
    );
}
