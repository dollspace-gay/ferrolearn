//! Divergence pin: ferrolearn's brute-regime `kneighbors` reproduces sklearn's
//! `parallel_on_Y` reduction with a SINGLE per-query max-heap built over ALL `n`
//! training points in index order. That is only correct for sklearn's
//! `chunks_n_threads == 1` (single Y-chunk OR `OMP_NUM_THREADS=1`). For
//! `n_samples > pairwise_dist_chunk_size` (256) sklearn splits Y into multiple
//! chunks and, on any machine with >= 2 hardware threads, builds one max-heap
//! PER chunk-thread and merges them in `_parallel_on_Y_synchronize`
//! (`sklearn/metrics/_pairwise_distances_reduction/_argkmin.pyx.tp:237-261`).
//! That per-chunk-then-merge reduction resolves an exact distance tie at the
//! k-th boundary DIFFERENTLY from a single global-index-order heap.
//!
//! This is the DEFAULT user-facing output: a default `KNeighborsClassifier`
//! (`metric='minkowski'`, `p=2`, `algorithm='auto'`) with `n_features > 15`
//! routes to `brute` (`sklearn/neighbors/_base.py:610-618`), whose auto
//! strategy is `parallel_on_Y` here, and on a normal multi-core box that is the
//! output every user sees — it is NOT an `OMP_NUM_THREADS=1` corner case.
//!
//! ## Live sklearn 1.5.2 / numpy 2.4.5 oracle (system python3), this fixture
//!
//! ```text
//! n=400, d=20, k=8.  X = zeros((400,20)); X[:,0] = 10.0 ;
//!   X[5,0]=1 ; X[250,0]=2 ; X[260,0]=3 ; X[390,0]=4 .   Q = zeros((1,20)).
//! y = arange(400) % 2 .   KNeighborsClassifier(n_neighbors=8).fit(X,y)
//!   ._fit_method                         -> 'brute'
//!   .kneighbors(Q, return_distance=True)[1][0]:
//!
//!   OMP_NUM_THREADS=1   -> [5, 250, 260, 390, 2, 4, 0, 6]
//!   OMP_NUM_THREADS>=2  -> [5, 250, 260, 390, 6, 2, 4, 3]   (STABLE for 2..28)
//! ```
//!
//! For `n=400`, `Y_n_chunks = ceil(400/256) = 2`, so `chunks_n_threads =
//! min(2, effective_n_threads)`. Hence the `>=2` value is the SAME for every
//! thread count from 2 to 28 (verified) — it is a stable, reproducible,
//! deterministic contract for any default multi-core sklearn install, NOT
//! thread-noise. The four points at index {0,2,3,4,6} are all at distance 10.0
//! (an exact tie); sklearn keeps the multiset {6,2,4,3}, ferrolearn keeps
//! {2,4,0,6}. The first four neighbors (the strict winners 5/250/260/390) and
//! all eight distances are identical; only the tie SET + ORDER at the k-boundary
//! diverge.
//!
//! ferrolearn (`brute_parallel_on_y`, single global-index-order heap,
//! `ferrolearn-neighbors/src/knn.rs`) returns the `OMP_NUM_THREADS=1` value
//! `[5, 250, 260, 390, 2, 4, 0, 6]`, which a default multi-core sklearn user
//! never sees. The builder's doc-comment claims the brute path "matches the
//! live sklearn 1.5.2 oracle ... bit-for-bit" and that for `n < 256` "there is
//! exactly one Y-chunk and one thread, so the result is fully deterministic" —
//! true, but the implementation extends that SINGLE-heap model to `n > 256`,
//! where sklearn's default is the MULTI-chunk merge.
//!
//! Expected values are the LIVE sklearn oracle, regenerated fresh, NOT copied
//! from ferrolearn (R-CHAR-3). Tracking: #2144.

use ferrolearn_core::traits::Fit;
use ferrolearn_neighbors::{Algorithm, KNeighborsClassifier};
use ndarray::{Array1, Array2};

/// Explicit (no-RNG) reconstruction of the oracle fixture.
///
/// `X`: 400x20, all zero except column 0 = 10.0, with four "sure winner"
/// overrides at indices 5/250/260/390 (distances 1/2/3/4 from the origin
/// query). Every other training point is at distance exactly 10.0 — a 396-way
/// exact tie, of which `k - 4 = 4` members fill the neighbor list.
fn fixture() -> (Array2<f64>, Array1<usize>, Array2<f64>) {
    let n = 400usize;
    let d = 20usize;
    let mut x = Array2::<f64>::zeros((n, d));
    for i in 0..n {
        x[[i, 0]] = 10.0;
    }
    for (idx, val) in [(5usize, 1.0), (250, 2.0), (260, 3.0), (390, 4.0)] {
        x[[idx, 0]] = val;
    }
    let y = Array1::from_iter(0..n);
    let q = Array2::<f64>::zeros((1, d));
    (x, y, q)
}

/// Divergence: brute-regime (`n_features > 15`) `kneighbors` at `n=400 > 256`
/// returns the `OMP_NUM_THREADS=1` single-heap tie resolution, but sklearn's
/// DEFAULT (multi-core, 2 Y-chunks) `KNeighborsClassifier.kneighbors()` returns
/// a different, stable, deterministic tie resolution.
///
/// sklearn DEFAULT (multi-core) order: [5, 250, 260, 390, 6, 2, 4, 3]
/// ferrolearn (single-heap)     order: [5, 250, 260, 390, 2, 4, 0, 6]
///
/// Tracking: #2144
#[test]
fn divergence_knn_multichunk_parallel_on_y_tie_set() {
    // This fixture (n=400 → 2 Y-chunks) pins the DEFAULT multi-core tie
    // resolution (effective_n_threads >= 2, each chunk its own thread). Under
    // `OMP_NUM_THREADS=1` sklearn itself collapses to the single-heap order
    // `[5, 250, 260, 390, 2, 4, 0, 6]` (which ferrolearn also returns and is
    // equally correct for that env), so skip rather than assert the multi-core
    // expectation in a single-thread environment.
    if std::env::var("OMP_NUM_THREADS").as_deref() == Ok("1") {
        eprintln!("skipped: OMP_NUM_THREADS=1 -> single-heap order (sklearn agrees)");
        return;
    }

    let (x, y, q) = fixture();
    let k = 8usize;

    let fitted = KNeighborsClassifier::<f64>::new()
        .with_n_neighbors(k)
        .with_algorithm(Algorithm::Auto)
        .fit(&x, &y)
        .unwrap();

    let (_dist, indices) = fitted.kneighbors(&q, Some(k)).unwrap();
    let got: Vec<usize> = indices.row(0).to_vec();

    // Live sklearn 1.5.2 / numpy 2.4.5, OMP_NUM_THREADS>=2 (default multi-core),
    // stable across thread counts 2..28:
    //   KNeighborsClassifier(8).fit(X,y).kneighbors(Q)[1][0]
    let sklearn_default_multicore: Vec<usize> = vec![5, 250, 260, 390, 6, 2, 4, 3];

    assert_eq!(
        got, sklearn_default_multicore,
        "n=400>256 brute: ferrolearn kneighbors {got:?} must equal sklearn's \
         DEFAULT multi-core .kneighbors() {sklearn_default_multicore:?} \
         (parallel_on_Y with 2 Y-chunks, stable for OMP_NUM_THREADS 2..28). \
         ferrolearn returns the single-heap / OMP_NUM_THREADS=1 tie resolution \
         [5, 250, 260, 390, 2, 4, 0, 6] instead, which a default multi-core \
         sklearn user never sees."
    );
}
