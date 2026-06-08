//! k-Nearest Neighbors classifier and regressor.
//!
//! This module provides [`KNeighborsClassifier`] and [`KNeighborsRegressor`],
//! which classify or predict target values based on the `k` nearest training
//! samples using Euclidean distance.
//!
//! # Algorithm Selection
//!
//! The [`Algorithm`] enum controls the spatial indexing strategy:
//!
//! - [`Algorithm::Auto`]: Automatically selects KD-Tree for dimensions <= 20,
//!   brute force otherwise.
//! - [`Algorithm::BruteForce`]: Always uses O(n) exhaustive search.
//! - [`Algorithm::KdTree`]: Always uses the KD-Tree spatial index.
//!
//! # Weighting
//!
//! The [`Weights`] enum controls how neighbor contributions are combined:
//!
//! - [`Weights::Uniform`]: All neighbors contribute equally.
//! - [`Weights::Distance`]: Closer neighbors contribute more (inverse distance).
//!
//! # Examples
//!
//! ```
//! use ferrolearn_neighbors::KNeighborsClassifier;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array1, Array2};
//!
//! let clf = KNeighborsClassifier::<f64>::new();
//! let x = Array2::from_shape_vec((6, 2), vec![
//!     0.0, 0.0, 0.5, 0.0, 0.0, 0.5,
//!     5.0, 5.0, 5.5, 5.0, 5.0, 5.5,
//! ]).unwrap();
//! let y = array![0, 0, 0, 1, 1, 1];
//!
//! let fitted = clf.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! ```
//!
//! ## REQ status
//!
//! Mirrors `sklearn.neighbors.KNeighborsClassifier`
//! (`sklearn/neighbors/_classification.py`) and `KNeighborsRegressor`
//! (`sklearn/neighbors/_regression.py`), both thin estimators over the shared
//! `KNeighborsMixin` / `NeighborsBase` machinery (`sklearn/neighbors/_base.py`).
//! See `.design/neighbors/knn.md`. `KNeighbors{Classifier,Regressor}` and their
//! fitted types are existing pub APIs re-exported via `pub use knn::{...}` in
//! `lib.rs`. Non-test consumers: `ferrolearn-python` `_RsKNeighborsClassifier`
//! (`ferrolearn-python/src/classifiers.rs`: `fit`/`predict`/`classes_`) and
//! `_RsKNeighborsRegressor` (`ferrolearn-python/src/extras.rs`, `py_regressor!`
//! macro: `fit`/`predict`); the in-crate `impl PipelineEstimator` for both; and
//! `graph.rs` (`FittedKNeighborsClassifier::kneighbors_graph` /
//! `FittedKNeighborsRegressor::kneighbors_graph` both call `self.kneighbors(...)`).
//! Cites use symbol anchors (ferrolearn) / `file:line` (sklearn 1.5.2). Live
//! oracle = installed sklearn 1.5.2, run from `/tmp`. Binary classification
//! (R-DEFER-2); honest underclaim (R-HONEST-3): a REQ is SHIPPED only with impl +
//! a NON-test consumer; `predict_proba`/`score` are value-correct (green guards
//! pass) but reach NO non-test consumer (the python bindings do not expose them),
//! so they are NOT-STARTED under #877.
//!
//! | REQ | Description | Status |
//! |-----|-------------|--------|
//! | REQ-1 | clf `predict` VALUE + smallest-label tie-break: per-row weighted vote, argmax first-max (`np.argmax`, `_classification.py:268`), uniform + distance. `fn predict` for `FittedKNeighborsClassifier` → `fn weighted_vote` over `fn class_score_vec` (argmax replaces on strict `>` so ties keep the smaller, earlier, sorted-`classes` label) mirrors `KNeighborsClassifier.predict` (`_classification.py:240-305`). Consumer: `_RsKNeighborsClassifier::predict` + `impl PipelineEstimator`. Green guards `green_classifier_predict_value_uniform_and_distance`, `green_classifier_tiebreak_smallest_label`. | SHIPPED |
//! | REQ-2 | clf `predict_proba` VALUE: normalized weighted class-vote shares in `classes_` order + zero-distance branch, mirroring `predict_proba` + `_get_weights` (`_classification.py:307`, `_base.py`). `pub fn predict_proba` (`class_score_vec` → `scores[ci]/total`) is value-correct (green `green_classifier_predict_proba_value`) but has NO non-test consumer — `_RsKNeighborsClassifier` exposes no `predict_proba` (the `predict_proba` at `classifiers.rs:404` belongs to `RsGaussianNB`). | NOT-STARTED (#877) |
//! | REQ-3 | clf `score` (mean accuracy), the `ClassifierMixin.score` analog. `pub fn score` (`correct/n` over `predict`) is value-correct (green `green_classifier_score_accuracy`) but has NO non-test consumer — `_RsKNeighborsClassifier` exposes no `score`. | NOT-STARTED (#877) |
//! | REQ-4 | reg `predict` VALUE, 1-D `y`, uniform + distance + zero-distance: weighted mean of neighbor targets, mirroring `KNeighborsRegressor.predict` (`_regression.py:229-270`). `fn predict` for `FittedKNeighborsRegressor` → `fn weighted_mean`. Consumer: `_RsKNeighborsRegressor::predict` (`py_regressor!`) + `impl PipelineEstimator`. Green guard `green_regressor_predict_value_uniform_and_distance`. | SHIPPED |
//! | REQ-5 | reg `score` R²: `RegressorMixin.score` = `r2_score(multioutput='uniform_average')`, constant-`y` → `1.0`/`-inf`. `pub fn score` → `pub(crate) fn r2_score` is value-correct but has NO non-test consumer — the `py_regressor!` macro (`extras.rs:17-58`) emits no `score` method, so `_RsKNeighborsRegressor` exposes none. | NOT-STARTED (#877) |
//! | REQ-6 | shared k-NN search VALUE + sklearn DEFAULT-backend exact-tie SET + ORDER (NOT algorithm-invariant): nearest-first `(distances, indices)` of shape `(n_queries, k)`. The f64 selection replicates the backend sklearn's DEFAULT `algorithm='auto'` user actually reaches — which is the `brute` `parallel_on_Y` strategy OR `kd_tree` `KDTree.query`, NEVER the internal `parallel_on_X` single-heap (the prior impl's bug). The auto rule (`_base.py:607-640`) for euclidean/p=2: `n_features > 15` OR `k >= n_samples // 2` → `brute` else `kd_tree`. Brute path = `fn brute_parallel_on_y` (the DOUBLE-HEAP transform of `_argkmin.pyx.tp`: push all train points in index order into heap `h1`, re-push `h1`'s heap-array slots `0..k` in heap order into `h2`, sort). kd_tree path = `crate::sk_kdtree::SkKdTree::{build,query}` (bit-exact `_recursive_build`/`init_node`/`find_node_split_dim` + `_query_single_depthfirst`/`min_rdist`, with `partition_node_indices` delegated to the bit-exact libstdc++ `std::nth_element` port `crate::introselect::nth_element` — the index layout that determines the kd_tree tie SET). Both backends reuse the same `NeighborsHeap` kernel `fn heap_push` (port `sklearn/utils/_heap.pyx:6-85`, ties at boundary REJECTED) + `fn simultaneous_sort` (port `sklearn/utils/_sorting.pyx:18-93`). `pub fn kneighbors` (both fitted types) → `pub(crate) fn kneighbors_impl` → per-row `fn find_neighbors` (routes brute/kd_tree); f32/other `F` retain the `(distance, index)` brute fallback (`kdtree::brute_force_knn`; Python binding is f64-only). Non-test consumer: `graph.rs` `FittedKNeighbors{Classifier,Regressor}::kneighbors_graph` (call `self.kneighbors(...)`). Green guards `green_kneighbors_value`, k>n `green_kneighbors_k_too_large_errors`, `test_knn_tie_break_lowest_index_all_algorithms`, `test_neighbors_heap_matches_sklearn_kneighbors_order`; tie pins `divergence_knn_tiebreak_*` (#2139, #2141) and the DEFAULT-order pin `divergence_kneighbors_order_differs_from_sklearn_default` (#2143, parallel_on_Y `[0,2,3,1,4]`). | SHIPPED |
//! | REQ-7 | `HasClasses` / `classes_`: sorted-unique class labels in lexicographic order (`_classification.py:120`). `impl HasClasses for FittedKNeighborsClassifier` (`classes()`/`n_classes()`). Non-test consumer: `_RsKNeighborsClassifier::classes_` getter (`classifiers.rs`) → `fitted.classes()`. In-tree `test_classifier_has_classes`. | SHIPPED |
//! | REQ-8 | fit-timing parity: `fit` no longer errors when `n_neighbors > n_samples` — sklearn `_fit` has no such check and defers the `n_neighbors <= n_samples_fit` `ValueError` to query time (`_base.py:828-832`), so `fit` succeeds and only `kneighbors`/`predict` error. BOTH `fn fit` methods validate only `n_neighbors == 0`; no fit-time `n_samples < n_neighbors` guard. Green `divergence_fit_does_not_error_when_n_neighbors_gt_n_samples` + in-module `test_classifier_fit_succeeds_when_k_gt_n`. | SHIPPED (#879) |
//! | REQ-9 | reg multi-output 2-D `y`: `_y.ndim > 1` → `(n_queries, n_outputs)` prediction (`_regression.py:253-270`). `FittedKNeighborsRegressor` stores `y_train: Array1<F>` and `impl Fit<Array2<F>, Array1<F>>` — 1-D `y` only; no 2-D surface. | NOT-STARTED (#875) |
//! | REQ-10 | constructor params + callable weights: `leaf_size=30`/`p=2`/`metric='minkowski'`/`metric_params=None`/`n_jobs=None` (`_classification.py:199-203`) and `weights` as a callable (`:190`). `KNeighbors{Classifier,Regressor}` have only `n_neighbors`/`algorithm`/`weights ∈ {Uniform, Distance}` — Euclidean-only, no callable variant. | NOT-STARTED (#876) |
//! | REQ-11 | PyO3 surface under-exposed: bind `predict_proba`/classifier-`score`/`weights`/`algorithm` on `_RsKNeighborsClassifier` and `weights` on `_RsKNeighborsRegressor`. The bindings exist but expose only fit/predict(/classes_); distance-weighting and `predict_proba` are unreachable from `import ferrolearn`. | NOT-STARTED (#877) |
//! | REQ-12 | ferray substrate: `knn.rs` imports `ndarray::{Array1, Array2}` + `num_traits::Float`, not `ferray-core` (R-SUBSTRATE). | NOT-STARTED (#878) |
//! | REQ-13 | non-finite input rejected: both `Fit::fit for KNeighborsClassifier` (X only — labels are integer `Array1<usize>`) and `Fit::fit for KNeighborsRegressor` (X AND float `Array1<F>` y) reject any NaN/+/-inf with `FerroError::InvalidParameter { name: "X"/"y", reason: "Input X/y contains NaN or infinity." }` AFTER the `n_neighbors == 0` constructor-param check but BEFORE the length checks and the spatial-index build — mirroring sklearn's `@_fit_context` `estimator._validate_params()` (`sklearn/base.py:1466`) running BEFORE the `fit` body's `_validate_data(X, y, ...)` finiteness check (`sklearn/neighbors/_base.py:475`, `force_all_finite=True` → `ValueError`). So a doubly-degenerate input (`n_neighbors=0` AND NaN X) raises on `n_neighbors`, matching the oracle (#2273). The QUERY path also guards: `fn predict` (clf + reg), `pub fn predict_proba`, and `pub(crate) fn kneighbors_impl` reject a non-finite query X with the same `InvalidParameter { name: "X" }` BEFORE the neighbor search, mirroring `KNeighborsMixin.kneighbors`'s `_validate_data(X, ..., reset=False)` (`sklearn/neighbors/_base.py:824`) — a NaN query previously PANICKED via an OOB neighbor index `y_train[usize::MAX]` (R-CODE-2 release-blocker, #2274). `.iter().any(\|v\| !v.is_finite())` catches NaN and Inf; the finite path is byte-identical (the guard never fires on finite input — all in-tree tests unchanged). Non-test consumers: the existing `Fit::fit` / `_RsKNeighborsClassifier` / `_RsKNeighborsRegressor` / pipeline `fit_pipeline` consumers. Verified vs the live sklearn 1.5.2 oracle (R-CHAR-3): `KNeighbors{Classifier,Regressor}().fit` raise `ValueError` for NaN/+inf/-inf in X (and NaN/inf in regressor y) — `tests/divergence_neighbors_nonfinite.rs::knn_*`; param-before-finiteness order `tests/divergence_neighbors_nonfinite_param_order.rs::knn_classifier_param_before_finiteness_order` (#2273); predict NaN/inf query → Err not panic `tests/divergence_neighbors_predict_nonfinite.rs::knn_predict_{nan,inf}_query_returns_err_not_panic` (#2274). | SHIPPED (#2272, #2273, #2274) |

use std::any::TypeId;

use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasClasses;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, PipelineEstimator};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rayon::prelude::*;

use crate::balltree::BallTree;
use crate::kdtree::{self, KdTree};
use crate::sk_kdtree::SkKdTree;

// ---------------------------------------------------------------------------
// Configuration enums
// ---------------------------------------------------------------------------

/// The algorithm used to compute nearest neighbors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Algorithm {
    /// Automatically select the best algorithm based on data characteristics.
    /// Uses KD-Tree for dimensions <= 15, ball tree for higher dimensions.
    Auto,
    /// Use brute-force exhaustive search (O(n) per query).
    BruteForce,
    /// Use a KD-Tree spatial index (O(log n) average per query for low dimensions).
    KdTree,
    /// Use a ball tree spatial index (handles high dimensions better than KD-Tree).
    BallTree,
}

/// The weighting scheme for neighbor contributions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Weights {
    /// All neighbors contribute equally (majority vote / simple mean).
    Uniform,
    /// Closer neighbors contribute more, weighted by inverse distance.
    /// If a query point exactly coincides with a training point (distance = 0),
    /// that point receives all weight.
    Distance,
}

// ---------------------------------------------------------------------------
// Helper: find k nearest neighbors
// ---------------------------------------------------------------------------

/// Which spatial index was built during fit.
pub(crate) enum SpatialIndex {
    None,
    KdTree(KdTree),
    BallTree(BallTree),
}

impl std::fmt::Debug for SpatialIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpatialIndex::None => write!(f, "None"),
            SpatialIndex::KdTree(t) => write!(f, "KdTree({t:?})"),
            SpatialIndex::BallTree(t) => write!(f, "BallTree({t:?})"),
        }
    }
}

/// Push a `(val, val_idx)` pair onto a fixed-size max-heap held as a
/// structure-of-arrays (`values` + `indices`).
///
/// Faithful Rust port of sklearn's `heap_push`
/// (`sklearn/utils/_heap.pyx:6-85`). The heap keeps the `size` SMALLEST values
/// seen, with `values[0]` the current maximum. A candidate is REJECTED when it
/// is not strictly closer than the root (`val >= values[0]`), so an
/// exact-distance tie at the boundary is rejected and the incumbent is kept —
/// this is what produces sklearn's tie-resolution. After replacing the root the
/// new value sifts down, the children comparison (`>=` on the left, strict `<`
/// for the swap test) copied verbatim from the upstream `noexcept nogil` kernel.
///
/// The math is total over finite, non-negative squared euclidean distances
/// (no `NaN`); no panic path. Specialised to `f64` because the dense f64
/// euclidean estimator surface (and the Python binding) is the contract here.
#[inline]
pub(crate) fn heap_push(
    values: &mut [f64],
    indices: &mut [usize],
    size: usize,
    val: f64,
    val_idx: usize,
) {
    // Reject if `val` should not be in the heap (ties at the boundary rejected).
    if val >= values[0] {
        return;
    }

    // Insert `val` at the root and sift it down.
    values[0] = val;
    indices[0] = val_idx;

    let mut current = 0usize;
    loop {
        let left = 2 * current + 1;
        let right = left + 1;

        // The `right >= size` and `values[left] >= values[right]` arms both
        // resolve to `left`/`break`, but on DISTINCT conditions transcribed
        // verbatim from sklearn `_heap.pyx:59-75`. Collapsing them would
        // obscure the faithful port; keep the upstream branch shape.
        #[allow(
            clippy::if_same_then_else,
            reason = "faithful 1:1 port of sklearn heap_push branch structure (_heap.pyx:59-75)"
        )]
        let swap = if left >= size {
            break;
        } else if right >= size {
            if values[left] > val {
                left
            } else {
                break;
            }
        } else if values[left] >= values[right] {
            if val < values[left] {
                left
            } else {
                break;
            }
        } else if val < values[right] {
            right
        } else {
            break;
        };

        values[current] = values[swap];
        indices[current] = indices[swap];
        current = swap;
    }

    values[current] = val;
    indices[current] = val_idx;
}

/// Swap entries `a` and `b` of both `values` and `indices` (sklearn
/// `dual_swap`, `sklearn/utils/_sorting.pyx:3-16`).
#[inline]
fn dual_swap(values: &mut [f64], indices: &mut [usize], a: usize, b: usize) {
    values.swap(a, b);
    indices.swap(a, b);
}

/// In-place recursive median-of-three quicksort that sorts `values` ascending,
/// permuting `indices` in lockstep.
///
/// Faithful Rust port of sklearn's `simultaneous_sort`
/// (`sklearn/utils/_sorting.pyx:18-93`). The small-array specialisations
/// (`size == 2` / `size == 3`) and the median-of-three pivot arrangement are
/// transcribed exactly. Per the upstream note at `_sorting.pyx:39-43` this does
/// NOT break ties by index — equal-distance entries keep their partition order — so
/// the tie ORDER is whatever the quicksort partition produces, matching
/// sklearn. The recursion uses index offsets in place of Cython's pointer
/// arithmetic (`values + pivot_idx + 1`) so the same buffer is mutated, which
/// is essential to reproduce the exact tie order.
pub(crate) fn simultaneous_sort(values: &mut [f64], indices: &mut [usize], size: usize) {
    if size <= 1 {
        // nothing to do
    } else if size == 2 {
        if values[0] > values[1] {
            dual_swap(values, indices, 0, 1);
        }
    } else if size == 3 {
        if values[0] > values[1] {
            dual_swap(values, indices, 0, 1);
        }
        if values[1] > values[2] {
            dual_swap(values, indices, 1, 2);
            if values[0] > values[1] {
                dual_swap(values, indices, 0, 1);
            }
        }
    } else {
        // Median-of-three pivot: smallest -> front, pivot value -> end.
        let mut pivot_idx = size / 2;
        if values[0] > values[size - 1] {
            dual_swap(values, indices, 0, size - 1);
        }
        if values[size - 1] > values[pivot_idx] {
            dual_swap(values, indices, size - 1, pivot_idx);
            if values[0] > values[size - 1] {
                dual_swap(values, indices, 0, size - 1);
            }
        }
        let pivot_val = values[size - 1];

        // Partition about the pivot.
        let mut store_idx = 0usize;
        for i in 0..size - 1 {
            if values[i] < pivot_val {
                dual_swap(values, indices, i, store_idx);
                store_idx += 1;
            }
        }
        dual_swap(values, indices, store_idx, size - 1);
        pivot_idx = store_idx;

        // Recurse on each side (offset slices = upstream pointer arithmetic).
        if pivot_idx > 1 {
            simultaneous_sort(
                &mut values[..pivot_idx],
                &mut indices[..pivot_idx],
                pivot_idx,
            );
        }
        if pivot_idx + 2 < size {
            simultaneous_sort(
                &mut values[pivot_idx + 1..],
                &mut indices[pivot_idx + 1..],
                size - pivot_idx - 1,
            );
        }
    }
}

/// sklearn brute `ArgKmin` DEFAULT strategy (`parallel_on_Y`) over
/// already-computed SQUARED euclidean distances.
///
/// This is the backend the DEFAULT user-facing `KNeighbors*.kneighbors()` /
/// `predict` reaches for the small-`n` / high-`k` regime (sklearn auto picks
/// `brute`, brute's auto strategy picks `parallel_on_Y`). For
/// `n_train < pairwise_dist_chunk_size` (256) there is exactly one Y-chunk and
/// one thread, so the result is fully deterministic across thread counts.
///
/// `parallel_on_Y` differs from `parallel_on_X` (the single-heap result) by a
/// per-chunk-heap-then-merge reduction, transcribed from
/// `sklearn/metrics/_pairwise_distances_reduction/_base.pyx.tp` (Y-chunking) +
/// `_argkmin.pyx.tp:237-261` (`_parallel_on_Y_synchronize`):
///
/// 1. Y-chunking (`_base.pyx.tp`): Y is split into `Y_n_chunks = ceil(n / 256)`
///    contiguous chunks of `pairwise_dist_chunk_size = 256` points
///    (`Y_n_samples_chunk = min(n, 256)`). The chunks are driven by
///    `prange(Y_n_chunks, schedule='static')` over
///    `chunks_n_threads = min(Y_n_chunks, effective_n_threads)` OpenMP threads
///    (`_base.pyx.tp:201-203`, `:312`). With libgomp `schedule='static'` (no
///    explicit chunk size) each thread is assigned ONE CONTIGUOUS BLOCK of
///    Y-chunks; the `Y_n_chunks` chunks split across `chunks_n_threads` threads
///    as near-equal contiguous blocks, the first
///    `Y_n_chunks % chunks_n_threads` threads getting one extra chunk. Each
///    thread accumulates ALL points in its block of chunks into ONE per-THREAD
///    heap `h1` of size `k` (init `+inf`/`-1`), pushing in INDEX order via
///    `heap_push`. For `n <= 256` there is exactly ONE chunk and one thread
///    spanning `0..n`, so this collapses to a single global heap (the
///    single-thread / `OMP_NUM_THREADS=1` result) — byte-identical to the prior
///    single-heap impl.
/// 2. `_parallel_on_Y_synchronize` (`_argkmin.pyx.tp:237-261`): the main heap
///    `h2` of size `k` (init `+inf`/`-1`); for each `thread_num` in
///    `0..chunks_n_threads` (THREAD-major, NOT chunk-major over `Y_n_chunks`),
///    for `jdx in 0..k` push `(h1.values[jdx], h1.indices[jdx])` — i.e. re-push
///    each thread's heap-array slots in heap order, NOT index order. Re-pushing
///    under-filled sentinel slots is a harmless no-op (the `val >= values[0]`
///    reject keeps `+inf`/`usize::MAX` out of `h2`). This per-thread-then-merge
///    is exactly why the tie SET/ORDER differs from a single global-index-order
///    heap once `n > 256`, AND why the SET depends on `effective_n_threads`
///    once `Y_n_chunks > effective_n_threads` (multiple chunks then share a
///    thread, so the boundary tie tracks the machine's thread count).
/// 3. `simultaneous_sort(h2)` ascending.
///
/// The result is intentionally `effective_n_threads`-dependent, mirroring
/// sklearn's own OpenMP-scheduling-dependent tie resolution on the same machine
/// (the same precedent as ferrolearn replicating numpy's CPU-dependent AVX2 sort
/// path): when `Y_n_chunks <= effective_n_threads` (`chunks_n_threads ==
/// Y_n_chunks`) each thread owns exactly one chunk and this reduces to the
/// chunk-major merge; when `Y_n_chunks > effective_n_threads` chunks are grouped
/// per thread and the boundary tie SET tracks the machine's thread count.
///
/// Returns `(indices, sorted_squared_distances)` in sklearn's exact default
/// brute k-NN order.
fn brute_parallel_on_y(sq_dist: &[f64], k: usize) -> (Vec<usize>, Vec<f64>) {
    const CHUNK_SIZE: usize = 256;
    let n = sq_dist.len();
    let eff = effective_n_threads(); // sklearn `_openmp_effective_n_threads(None)`
    let m = n.div_ceil(CHUNK_SIZE); // Y_n_chunks
    let t = m.min(eff); // chunks_n_threads = min(Y_n_chunks, effective_n_threads)

    // libgomp `schedule='static'` (no chunk size): distribute the `m` chunks
    // across `t` threads as contiguous blocks; the first `r` threads get one
    // extra chunk.
    let q = m / t;
    let r = m % t;

    // Main heap (init `+inf`/`-1`), filled by merging the per-thread heaps.
    let mut h2v = vec![f64::INFINITY; k];
    let mut h2i = vec![usize::MAX; k];

    let mut chunk_cursor = 0usize;
    for thread in 0..t {
        let n_chunks_this = q + usize::from(thread < r);
        let c_lo = chunk_cursor;
        let c_hi = chunk_cursor + n_chunks_this;
        chunk_cursor = c_hi;

        let idx_lo = c_lo * CHUNK_SIZE;
        let idx_hi = (c_hi * CHUNK_SIZE).min(n);

        // Per-THREAD heap: push this thread's contiguous index range in index
        // order. `idx_lo + off` is the GLOBAL training index (the heap index).
        let mut hv = vec![f64::INFINITY; k];
        let mut hi = vec![usize::MAX; k];
        for (off, &d) in sq_dist[idx_lo..idx_hi].iter().enumerate() {
            heap_push(&mut hv, &mut hi, k, d, idx_lo + off);
        }

        // Merge into the main heap (thread-major): re-push this thread's
        // heap-array slots jdx=0..k-1 in heap order.
        for jdx in 0..k {
            heap_push(&mut h2v, &mut h2i, k, hv[jdx], hi[jdx]);
        }
    }

    simultaneous_sort(&mut h2v, &mut h2i, k);
    (h2i, h2v)
}

/// sklearn brute `ArgKmin` `parallel_on_X` strategy over already-computed
/// SQUARED euclidean distances.
///
/// sklearn's `auto` strategy resolver (`_base.pyx.tp:183-196`) switches the
/// brute `ArgKmin.compute(X=queries, Y=training)` reduction from the default
/// `parallel_on_Y` to `parallel_on_X` whenever `n_samples_Y < n_samples_X`
/// (i.e. `n_train < n_queries`) OR `4 * chunk_size * effective_n_threads <
/// n_samples_X`. Under `parallel_on_X` each query owns its OWN heap and the
/// Y-chunks are pushed SEQUENTIALLY in global training-index order `0..n` into
/// that single heap, finalized by `simultaneous_sort`
/// (`_parallel_on_X_prange_iter_finalize`). So for one query this is exactly
/// the SINGLE global-heap select over all of Y in index order — NOT the
/// per-chunk-then-merge double-heap of [`brute_parallel_on_y`]. The SET is the
/// same as `parallel_on_Y`; only the exact-distance tie ORDER at the k-th
/// boundary differs.
///
/// Returns `(indices, sorted_squared_distances)`.
fn brute_parallel_on_x(sq_dist: &[f64], k: usize) -> (Vec<usize>, Vec<f64>) {
    let mut hv = vec![f64::INFINITY; k];
    let mut hi = vec![usize::MAX; k];
    for (i, &d) in sq_dist.iter().enumerate() {
        heap_push(&mut hv, &mut hi, k, d, i);
    }
    simultaneous_sort(&mut hv, &mut hi, k);
    (hi, hv)
}

/// Resolve sklearn's brute `ArgKmin` `auto` reduction strategy for a whole
/// `kneighbors` call (`_base.pyx.tp:183-196`).
///
/// For `kneighbors`, sklearn calls `ArgKmin.compute(X=queries, Y=training,
/// strategy="auto")` (`sklearn/neighbors/_base.py:849-851`), so
/// `n_samples_X = n_queries` and `n_samples_Y = n_train`. With
/// `chunk_size = 256` the `auto` resolver picks `parallel_on_X` when
/// `n_samples_Y < n_samples_X` OR `4 * chunk_size * effective_n_threads <
/// n_samples_X`, else `parallel_on_Y`. The decision depends on the TOTAL query
/// count, which only the per-call caller knows — hence it is computed once and
/// threaded into [`find_neighbors`] via a flag.
fn brute_use_parallel_on_x(n_train: usize, n_queries: usize) -> bool {
    const CHUNK_SIZE: usize = 256;
    n_train < n_queries || 4 * CHUNK_SIZE * effective_n_threads() < n_queries
}

/// sklearn `_openmp_effective_n_threads(None)`
/// (`sklearn/utils/_openmp_helpers.pyx`), which determines `chunks_n_threads`
/// for the brute `parallel_on_Y` reduction and therefore the OpenMP-scheduling
/// tie SET at the k-boundary (see [`brute_parallel_on_y`]).
///
/// Replicates sklearn exactly: if `OMP_NUM_THREADS` is set, `omp_get_max_threads`
/// honours it (first comma token); otherwise the value is
/// `min(omp_get_max_threads(), cpu_count(only_physical_cores=True))`, which on a
/// machine whose OMP default is the logical core count (>= physical) equals the
/// physical core count. The output is therefore intentionally machine-dependent,
/// mirroring sklearn's own behaviour on the same box. Never panics — every parse
/// failure degrades to the logical-parallelism fallback.
fn effective_n_threads() -> usize {
    // `OMP_NUM_THREADS` override (omp_get_max_threads honours it; first token).
    if let Ok(s) = std::env::var("OMP_NUM_THREADS")
        && let Some(tok) = s.split(',').next()
        && let Ok(v) = tok.trim().parse::<usize>()
        && v > 0
    {
        return v;
    }
    // Else physical core count (joblib `cpu_count(only_physical_cores=True)`).
    if let Some(p) = physical_cores_from_cpuinfo()
        && p > 0
    {
        return p;
    }
    // Fallback: logical parallelism.
    std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(1)
}

/// Count physical cores on Linux as the number of unique `(physical id, core id)`
/// pairs in `/proc/cpuinfo` (joblib's `only_physical_cores=True` heuristic).
///
/// Returns `None` (so the caller falls back to logical parallelism) when
/// `/proc/cpuinfo` is unreadable or lacks the `physical id` / `core id` fields
/// (some CPUs / VMs). Never panics.
fn physical_cores_from_cpuinfo() -> Option<usize> {
    use std::collections::HashSet;

    let text = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    let mut pairs: HashSet<(String, String)> = HashSet::new();
    let mut cur_phys: Option<String> = None;
    let mut cur_core: Option<String> = None;
    let mut saw_field = false;

    for line in text.lines() {
        if line.trim().is_empty() {
            // End of a processor block — record its (physical id, core id) pair.
            if let (Some(p), Some(c)) = (cur_phys.take(), cur_core.take()) {
                pairs.insert((p, c));
            }
            cur_phys = None;
            cur_core = None;
            continue;
        }
        if let Some((key, val)) = line.split_once(':') {
            let key = key.trim();
            let val = val.trim().to_string();
            if key == "physical id" {
                cur_phys = Some(val);
                saw_field = true;
            } else if key == "core id" {
                cur_core = Some(val);
                saw_field = true;
            }
        }
    }
    // Trailing block without a terminating blank line.
    if let (Some(p), Some(c)) = (cur_phys, cur_core) {
        pairs.insert((p, c));
    }

    if !saw_field || pairs.is_empty() {
        return None;
    }
    Some(pairs.len())
}

/// Find the k nearest neighbors of a query point.
///
/// Returns `(index, distance)` pairs nearest-first, where `distance` is the
/// (non-squared) euclidean distance — preserving the contract of
/// `kdtree::brute_force_knn` so `weights='distance'` weighting stays correct.
///
/// # f64 path — sklearn's ACTUAL default backend (NOT algorithm-invariant)
///
/// For `F == f64` (the Python-binding surface, dense euclidean `minkowski`
/// `p=2`) the SELECTION replicates the backend sklearn's DEFAULT
/// `algorithm='auto'` user actually reaches, which is NOT the single-heap
/// `parallel_on_X` result and is NOT algorithm-invariant — sklearn's `kd_tree`
/// and `brute` backends genuinely return different tie SETs/ORDERS on exact
/// distance ties (#2143). The auto rule (`sklearn/neighbors/_base.py:607-640`)
/// selects, for `p=2` low-dim:
///
/// - `n_features > 15` OR `k >= n_samples // 2` -> **brute**, whose `auto`
///   strategy (`_base.pyx.tp:183-196`) is `parallel_on_Y` (`brute_parallel_on_y`
///   — the double-heap transform of `_argkmin.pyx.tp`) UNLESS the caller's
///   `use_parallel_on_x` flag is set (`n_train < n_queries` OR
///   `4*256*effective_n_threads < n_queries`), in which case the single global
///   index-order heap `brute_parallel_on_x` is used.
/// - else -> **kd_tree**, a single-tree depth-first `KDTree.query`
///   ([`crate::sk_kdtree::SkKdTree`]), whose leaf push order (hence tie SET)
///   follows the bit-exact `std::nth_element` build partition.
///
/// Both backends push SQUARED euclidean distances onto the same
/// `NeighborsHeap` kernel (`heap_push` + `simultaneous_sort`); squared keeps
/// exact ties exact and order-identical to euclidean. The result matches the
/// live sklearn 1.5.2 oracle for a float64 query array (the dtype/contiguity
/// that makes sklearn route through the fast backends) bit-for-bit, including
/// the #2143 default-order pin and the #2139/#2141 unique-resolution fixtures.
///
/// The KDTree is rebuilt per query row here. This is wasteful but bit-correct;
/// the auto choice depends only on `(n_samples, n_features, k)` so it is
/// identical for every row.
///
/// # f32 / other `F` fallback
///
/// For non-`f64` `F` the existing stable `(distance, index)` brute top-k
/// (`kdtree::brute_force_knn`) is used. The Python binding is f64-only, so the
/// f64 backends cover the entire `import ferrolearn` surface; the f32 fallback
/// is a Rust-only convenience whose exact-tie order is NOT guaranteed to match
/// sklearn (sklearn computes f32 KNN through distinct `*32` kernels).
fn find_neighbors<F: Float + Send + Sync + 'static>(
    data: &Array2<F>,
    query_row: &[F],
    k: usize,
    _index: &SpatialIndex,
    use_parallel_on_x: bool,
) -> Vec<(usize, F)> {
    let n_samples = data.nrows();
    let k_clamped = k.min(n_samples);
    if k_clamped == 0 {
        return Vec::new();
    }

    // f64 path: route to the backend sklearn's default `auto` actually uses.
    if TypeId::of::<F>() == TypeId::of::<f64>() {
        let n_features = data.ncols();

        // sklearn auto rule (`_base.py:607-640`) for euclidean / p=2:
        //   n_features > 15 OR k >= n_samples // 2  -> brute (parallel_on_Y)
        //   else                                    -> kd_tree (KDTree.query)
        // `n` is n_samples_fit, the same `n` sklearn uses; `k` is the requested
        // n_neighbors (NOT the clamped value — the caller guards k > n).
        let use_brute = n_features > 15 || k >= n_samples / 2;

        // Materialize an f64 view of the query row.
        let q64: Vec<f64> = query_row
            .iter()
            .map(|&v| v.to_f64().unwrap_or(0.0))
            .collect();

        let (hi, hv) = if use_brute {
            // Squared euclidean distance for every training point, index order.
            let sq_dist: Vec<f64> = (0..n_samples)
                .map(|i| {
                    let mut acc = 0.0f64;
                    for j in 0..n_features {
                        let dij = data[[i, j]].to_f64().unwrap_or(0.0) - q64[j];
                        acc += dij * dij;
                    }
                    acc
                })
                .collect();
            // sklearn `ArgKmin` auto strategy (`_base.pyx.tp:183-196`):
            // `parallel_on_X` when `n_train < n_queries` OR
            // `4*256*effective_n_threads < n_queries` (computed by the caller,
            // which knows the total query count), else `parallel_on_Y`.
            let (hi, sq) = if use_parallel_on_x {
                brute_parallel_on_x(&sq_dist, k_clamped)
            } else {
                brute_parallel_on_y(&sq_dist, k_clamped)
            };
            // brute returns squared distances; convert to euclidean.
            let dists: Vec<f64> = sq.into_iter().map(|s| s.max(0.0).sqrt()).collect();
            (hi, dists)
        } else {
            // kd_tree: build a bit-exact sklearn KDTree on an f64 copy and run
            // the single-tree depth-first query. SkKdTree::query already
            // returns euclidean (non-squared) distances.
            let data64: Array2<f64> = data.mapv(|v| v.to_f64().unwrap_or(0.0));
            let tree = SkKdTree::build(&data64);
            tree.query(&q64, k_clamped)
        };

        return hi
            .into_iter()
            .zip(hv)
            .map(|(idx, dist)| {
                let d = F::from(dist).unwrap_or_else(F::zero);
                (idx, d)
            })
            .collect();
    }

    // f32 / other F: existing stable (distance, index) brute fallback.
    kdtree::brute_force_knn(data, query_row, k)
}

/// Build the appropriate spatial index based on algorithm setting and dimensionality.
fn build_spatial_index<F: Float + Send + Sync + 'static>(
    algorithm: Algorithm,
    data: &Array2<F>,
) -> SpatialIndex {
    let n_features = data.ncols();
    match algorithm {
        Algorithm::Auto => {
            if n_features <= 15 {
                SpatialIndex::KdTree(KdTree::build(data))
            } else {
                SpatialIndex::BallTree(BallTree::build(data))
            }
        }
        Algorithm::KdTree => SpatialIndex::KdTree(KdTree::build(data)),
        Algorithm::BallTree => SpatialIndex::BallTree(BallTree::build(data)),
        Algorithm::BruteForce => SpatialIndex::None,
    }
}

// ---------------------------------------------------------------------------
// KNeighborsClassifier
// ---------------------------------------------------------------------------

/// k-Nearest Neighbors classifier.
///
/// Classifies samples by majority vote of the `k` nearest training
/// samples using Euclidean distance.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
///
/// # Examples
///
/// ```
/// use ferrolearn_neighbors::KNeighborsClassifier;
/// use ferrolearn_core::{Fit, Predict};
/// use ndarray::{array, Array2};
///
/// let clf = KNeighborsClassifier::<f64>::new().with_n_neighbors(3);
/// let x = Array2::from_shape_vec((6, 2), vec![
///     0.0, 0.0, 0.5, 0.0, 0.0, 0.5,
///     5.0, 5.0, 5.5, 5.0, 5.0, 5.5,
/// ]).unwrap();
/// let y = array![0, 0, 0, 1, 1, 1];
///
/// let fitted = clf.fit(&x, &y).unwrap();
/// let preds = fitted.predict(&x).unwrap();
/// assert_eq!(preds.len(), 6);
/// ```
#[derive(Debug, Clone)]
pub struct KNeighborsClassifier<F> {
    /// Number of neighbors to use for classification.
    pub n_neighbors: usize,
    /// The algorithm to use for neighbor search.
    pub algorithm: Algorithm,
    /// The weighting scheme for neighbor contributions.
    pub weights: Weights,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float> KNeighborsClassifier<F> {
    /// Create a new `KNeighborsClassifier` with default settings.
    ///
    /// Defaults: `n_neighbors = 5`, `algorithm = Auto`, `weights = Uniform`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_neighbors: 5,
            algorithm: Algorithm::Auto,
            weights: Weights::Uniform,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the number of neighbors.
    #[must_use]
    pub fn with_n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.n_neighbors = n_neighbors;
        self
    }

    /// Set the algorithm for neighbor search.
    #[must_use]
    pub fn with_algorithm(mut self, algorithm: Algorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Set the weighting scheme.
    #[must_use]
    pub fn with_weights(mut self, weights: Weights) -> Self {
        self.weights = weights;
        self
    }
}

impl<F: Float> Default for KNeighborsClassifier<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted k-Nearest Neighbors classifier.
///
/// Stores the training data and an optional KD-Tree spatial index.
/// Implements [`Predict`] to classify new samples.
#[derive(Debug)]
pub struct FittedKNeighborsClassifier<F> {
    /// Training feature data.
    x_train: Array2<F>,
    /// Training labels.
    y_train: Array1<usize>,
    /// Number of neighbors to use.
    n_neighbors: usize,
    /// Weighting scheme.
    weights: Weights,
    /// Spatial index (KD-Tree, Ball Tree, or None for brute force).
    spatial_index: SpatialIndex,
    /// Sorted unique class labels.
    classes: Vec<usize>,
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<usize>> for KNeighborsClassifier<F> {
    type Fitted = FittedKNeighborsClassifier<F>;
    type Error = FerroError;

    /// Fit the classifier by storing the training data and optionally
    /// building a KD-Tree spatial index.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of samples in
    /// `x` and `y` differ.
    /// Returns [`FerroError::InvalidParameter`] if `n_neighbors` is zero.
    ///
    /// `n_neighbors > n_samples` is NOT rejected at fit time (matching
    /// sklearn, which defers the `n_neighbors <= n_samples_fit` check to
    /// query time); it surfaces from `kneighbors`/`predict` instead.
    fn fit(
        &self,
        x: &Array2<F>,
        y: &Array1<usize>,
    ) -> Result<FittedKNeighborsClassifier<F>, FerroError> {
        // sklearn validates CONSTRUCTOR PARAMS before the data: the
        // `@_fit_context` decorator runs `estimator._validate_params()`
        // (`sklearn/base.py:1466`) BEFORE the wrapped `fit` body's
        // `self._validate_data(X, y, ...)` finiteness check
        // (`:1472` → `NeighborsBase._fit`, `sklearn/neighbors/_base.py:475`).
        // So for a doubly-degenerate input (invalid `n_neighbors` AND NaN/inf in
        // X), sklearn raises `InvalidParameterError` naming `n_neighbors`, NOT the
        // data-finiteness `ValueError`. The `n_neighbors == 0` constructor-param
        // check therefore precedes the finiteness guard. (#2273)
        if self.n_neighbors == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_neighbors".into(),
                reason: "must be at least 1".into(),
            });
        }

        // sklearn `_validate_data` keeps the default `force_all_finite=True`, so
        // `check_array` rejects any NaN or +/-inf in X with a `ValueError` BEFORE
        // the index is built, but AFTER param validation (above) and BEFORE the
        // data-shape/class checks (below). `.iter().any(|v| !v.is_finite())`
        // rejects both NaN and Inf (bounds-safe, no panic, R-CODE-2). `y` is
        // `Array1<usize>` (integer labels), always finite, so only X is guarded
        // here. (#2272, reordered #2273)
        if x.iter().any(|v| !v.is_finite()) {
            return Err(FerroError::InvalidParameter {
                name: "X".into(),
                reason: "Input X contains NaN or infinity.".into(),
            });
        }

        let n_samples = x.nrows();

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        // sklearn does NOT validate n_neighbors vs n_samples at fit time
        // (`NeighborsBase._fit` has no such check); the
        // `n_neighbors <= n_samples_fit` test is deferred to query time in
        // `KNeighborsMixin.kneighbors` (sklearn/neighbors/_base.py:828-832).
        // ferrolearn mirrors this: `kneighbors_impl` already enforces
        // `n_neighbors > x_train.nrows()` at predict/kneighbors time, so no
        // fit-time guard is added here (#874).

        // Determine unique classes.
        let mut classes: Vec<usize> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();

        // Build spatial index.
        let spatial_index = build_spatial_index(self.algorithm, x);

        Ok(FittedKNeighborsClassifier {
            x_train: x.clone(),
            y_train: y.clone(),
            n_neighbors: self.n_neighbors,
            weights: self.weights,
            spatial_index,
            classes,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedKNeighborsClassifier<F> {
    type Output = Array1<usize>;
    type Error = FerroError;

    /// Predict class labels for the given feature matrix.
    ///
    /// For each sample, finds the `k` nearest neighbors in the training
    /// data and returns the majority class (with optional distance weighting).
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features
    /// does not match the training data.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<usize>, FerroError> {
        let n_features = x.ncols();
        let train_features = self.x_train.ncols();

        if n_features != train_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![train_features],
                actual: vec![n_features],
                context: "number of features must match training data".into(),
            });
        }

        // Query-time finiteness guard. sklearn `KNeighborsMixin.kneighbors`
        // re-validates the query X with `self._validate_data(X, ..., reset=False)`
        // (`sklearn/neighbors/_base.py:824`), keeping `force_all_finite=True`, so
        // a NaN/inf query raises `ValueError` BEFORE the neighbor search. Without
        // this guard a NaN query flows into the kd_tree/heap search and yields an
        // out-of-bounds neighbor index (`usize::MAX`), then `y_train[idx]` PANICS
        // (R-CODE-2 violation). Return a clean `Err` instead. (#2274)
        if x.iter().any(|v| !v.is_finite()) {
            return Err(FerroError::InvalidParameter {
                name: "X".into(),
                reason: "Input X contains NaN or infinity.".into(),
            });
        }

        // Query-time `n_neighbors <= n_samples_fit` guard, deferred from fit
        // (sklearn/neighbors/_base.py:828-838 raises ValueError here). Routes
        // through the same `InsufficientSamples` error as `kneighbors_impl`
        // (REQ-6 guard `green_kneighbors_k_too_large_errors`) so both query
        // paths raise identically rather than silently clamping `k` to the
        // available neighbors via `find_neighbors`.
        let n_samples_fit = self.x_train.nrows();
        if self.n_neighbors > n_samples_fit {
            return Err(FerroError::InsufficientSamples {
                required: self.n_neighbors,
                actual: n_samples_fit,
                context: "n_neighbors exceeds number of training samples".into(),
            });
        }

        let n_samples = x.nrows();

        // Brute `ArgKmin` auto strategy resolved once for the whole call.
        let use_parallel_on_x = brute_use_parallel_on_x(n_samples_fit, n_samples);

        // Use a threshold to avoid Rayon overhead on small inputs.
        const PAR_THRESHOLD: usize = 256;

        let predictions_vec: Vec<usize> = if n_samples >= PAR_THRESHOLD {
            (0..n_samples)
                .into_par_iter()
                .map(|i| {
                    let query: Vec<F> = (0..n_features).map(|j| x[[i, j]]).collect();
                    let neighbors = find_neighbors(
                        &self.x_train,
                        &query,
                        self.n_neighbors,
                        &self.spatial_index,
                        use_parallel_on_x,
                    );
                    self.weighted_vote(&neighbors)
                })
                .collect()
        } else {
            (0..n_samples)
                .map(|i| {
                    let query: Vec<F> = (0..n_features).map(|j| x[[i, j]]).collect();
                    let neighbors = find_neighbors(
                        &self.x_train,
                        &query,
                        self.n_neighbors,
                        &self.spatial_index,
                        use_parallel_on_x,
                    );
                    self.weighted_vote(&neighbors)
                })
                .collect()
        };

        Ok(Array1::from_vec(predictions_vec))
    }
}

impl<F: Float + Send + Sync + 'static> FittedKNeighborsClassifier<F> {
    /// Per-class weighted vote sums for one set of neighbors. Returned in
    /// `self.classes` order so the caller can normalize, argmax, etc.
    fn class_score_vec(&self, neighbors: &[(usize, F)]) -> Vec<F> {
        let mut scores = vec![F::zero(); self.classes.len()];
        let eps = F::from(1e-15).unwrap();
        // Map class label → position in self.classes for O(log n) lookup.
        let class_idx = |label: usize| -> usize {
            self.classes
                .binary_search(&label)
                .expect("label not in fitted classes")
        };

        match self.weights {
            Weights::Uniform => {
                for &(idx, _) in neighbors {
                    let ci = class_idx(self.y_train[idx]);
                    scores[ci] = scores[ci] + F::one();
                }
            }
            Weights::Distance => {
                let has_zero_dist = neighbors.iter().any(|&(_, d)| d < eps);
                if has_zero_dist {
                    for &(idx, d) in neighbors {
                        if d < eps {
                            let ci = class_idx(self.y_train[idx]);
                            scores[ci] = scores[ci] + F::one();
                        }
                    }
                } else {
                    for &(idx, d) in neighbors {
                        let ci = class_idx(self.y_train[idx]);
                        scores[ci] = scores[ci] + F::one() / d;
                    }
                }
            }
        }
        scores
    }

    /// Perform a (possibly weighted) majority vote among neighbors.
    /// Tie-break by smallest class label (sklearn parity).
    fn weighted_vote(&self, neighbors: &[(usize, F)]) -> usize {
        let scores = self.class_score_vec(neighbors);
        let mut best_idx = 0usize;
        let mut best_score = scores[0];
        for (i, &s) in scores.iter().enumerate().skip(1) {
            if s > best_score {
                best_score = s;
                best_idx = i;
            }
            // Equal scores → prefer the smaller class label (= earlier
            // index in self.classes since classes is sorted).
        }
        self.classes[best_idx]
    }

    /// Predict class probabilities for the given feature matrix.
    ///
    /// For each sample, finds the `k` nearest neighbors and returns the
    /// normalized (weighted) class vote shares, with classes laid out in
    /// the order of [`HasClasses::classes`].
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does
    /// not match the training data.
    pub fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features = x.ncols();
        let train_features = self.x_train.ncols();
        if n_features != train_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![train_features],
                actual: vec![n_features],
                context: "number of features must match training data".into(),
            });
        }

        // Query-time finiteness guard (see `predict`): sklearn re-validates the
        // query X (`_validate_data(..., reset=False)`,
        // `sklearn/neighbors/_base.py:824`) so a NaN/inf query raises before the
        // neighbor search rather than producing an OOB index panic. (#2274)
        if x.iter().any(|v| !v.is_finite()) {
            return Err(FerroError::InvalidParameter {
                name: "X".into(),
                reason: "Input X contains NaN or infinity.".into(),
            });
        }

        // Query-time `n_neighbors <= n_samples_fit` guard (see `predict`);
        // sklearn/neighbors/_base.py:828-838 raises ValueError here rather than
        // clamping `k`.
        let n_samples_fit = self.x_train.nrows();
        if self.n_neighbors > n_samples_fit {
            return Err(FerroError::InsufficientSamples {
                required: self.n_neighbors,
                actual: n_samples_fit,
                context: "n_neighbors exceeds number of training samples".into(),
            });
        }

        let n_samples = x.nrows();
        let n_classes = self.classes.len();
        let use_parallel_on_x = brute_use_parallel_on_x(n_samples_fit, n_samples);
        let mut proba = Array2::<F>::zeros((n_samples, n_classes));

        for i in 0..n_samples {
            let query: Vec<F> = (0..n_features).map(|j| x[[i, j]]).collect();
            let neighbors = find_neighbors(
                &self.x_train,
                &query,
                self.n_neighbors,
                &self.spatial_index,
                use_parallel_on_x,
            );
            let scores = self.class_score_vec(&neighbors);
            let total: F = scores.iter().copied().fold(F::zero(), |a, b| a + b);
            if total > F::zero() {
                for ci in 0..n_classes {
                    proba[[i, ci]] = scores[ci] / total;
                }
            } else {
                // No neighbors / all zero weights — fall back to uniform.
                let u = F::one() / F::from(n_classes).unwrap();
                for ci in 0..n_classes {
                    proba[[i, ci]] = u;
                }
            }
        }
        Ok(proba)
    }

    /// Mean accuracy on the given test data and labels.
    ///
    /// Equivalent to sklearn's `ClassifierMixin.score`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x.nrows() != y.len()` or
    /// the feature count does not match the training data.
    pub fn score(&self, x: &Array2<F>, y: &Array1<usize>) -> Result<F, FerroError> {
        if x.nrows() != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows()],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }
        let preds = self.predict(x)?;
        let n = y.len();
        if n == 0 {
            return Ok(F::zero());
        }
        let correct = preds.iter().zip(y.iter()).filter(|(p, t)| p == t).count();
        Ok(F::from(correct).unwrap() / F::from(n).unwrap())
    }
}

impl<F: Float + Send + Sync + 'static> HasClasses for FittedKNeighborsClassifier<F> {
    fn classes(&self) -> &[usize] {
        &self.classes
    }

    fn n_classes(&self) -> usize {
        self.classes.len()
    }
}

// Pipeline integration.
impl<F: Float + ToPrimitive + FromPrimitive + Send + Sync + 'static> PipelineEstimator<F>
    for KNeighborsClassifier<F>
{
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        // Convert float labels to usize.
        let y_usize: Array1<usize> = y.mapv(|v| v.to_usize().unwrap_or(0));
        let fitted = self.fit(x, &y_usize)?;
        Ok(Box::new(FittedKNeighborsClassifierPipeline(fitted)))
    }
}

/// Wrapper for pipeline integration that converts predictions to float.
struct FittedKNeighborsClassifierPipeline<F: Float + Send + Sync + 'static>(
    FittedKNeighborsClassifier<F>,
);

// Safety: FittedKNeighborsClassifier<F> is Send + Sync because all its
// fields (Array2<F>, Array1<usize>, usize, Weights, Option<KdTree>, Vec<usize>)
// are Send + Sync.
unsafe impl<F: Float + Send + Sync + 'static> Send for FittedKNeighborsClassifierPipeline<F> {}
unsafe impl<F: Float + Send + Sync + 'static> Sync for FittedKNeighborsClassifierPipeline<F> {}

impl<F: Float + ToPrimitive + FromPrimitive + Send + Sync + 'static> FittedPipelineEstimator<F>
    for FittedKNeighborsClassifierPipeline<F>
{
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        let preds = self.0.predict(x)?;

        Ok(preds.mapv(|v| F::from_usize(v).unwrap_or_else(F::nan)))
    }
}

// ---------------------------------------------------------------------------
// KNeighborsRegressor
// ---------------------------------------------------------------------------

/// k-Nearest Neighbors regressor.
///
/// Predicts target values as the (weighted) mean of the `k` nearest
/// training samples' target values using Euclidean distance.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
///
/// # Examples
///
/// ```
/// use ferrolearn_neighbors::KNeighborsRegressor;
/// use ferrolearn_core::{Fit, Predict};
/// use ndarray::{array, Array2};
///
/// let reg = KNeighborsRegressor::<f64>::new().with_n_neighbors(3);
/// let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
/// let y = array![2.0, 4.0, 6.0, 8.0, 10.0];
///
/// let fitted = reg.fit(&x, &y).unwrap();
/// let preds = fitted.predict(&x).unwrap();
/// assert_eq!(preds.len(), 5);
/// ```
#[derive(Debug, Clone)]
pub struct KNeighborsRegressor<F> {
    /// Number of neighbors to use for regression.
    pub n_neighbors: usize,
    /// The algorithm to use for neighbor search.
    pub algorithm: Algorithm,
    /// The weighting scheme for neighbor contributions.
    pub weights: Weights,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float> KNeighborsRegressor<F> {
    /// Create a new `KNeighborsRegressor` with default settings.
    ///
    /// Defaults: `n_neighbors = 5`, `algorithm = Auto`, `weights = Uniform`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_neighbors: 5,
            algorithm: Algorithm::Auto,
            weights: Weights::Uniform,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the number of neighbors.
    #[must_use]
    pub fn with_n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.n_neighbors = n_neighbors;
        self
    }

    /// Set the algorithm for neighbor search.
    #[must_use]
    pub fn with_algorithm(mut self, algorithm: Algorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Set the weighting scheme.
    #[must_use]
    pub fn with_weights(mut self, weights: Weights) -> Self {
        self.weights = weights;
        self
    }
}

impl<F: Float> Default for KNeighborsRegressor<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted k-Nearest Neighbors regressor.
///
/// Stores the training data and an optional KD-Tree spatial index.
/// Implements [`Predict`] to predict target values for new samples.
#[derive(Debug)]
pub struct FittedKNeighborsRegressor<F> {
    /// Training feature data.
    x_train: Array2<F>,
    /// Training target values.
    y_train: Array1<F>,
    /// Number of neighbors to use.
    n_neighbors: usize,
    /// Weighting scheme.
    weights: Weights,
    /// Spatial index (KD-Tree, Ball Tree, or None for brute force).
    spatial_index: SpatialIndex,
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<F>> for KNeighborsRegressor<F> {
    type Fitted = FittedKNeighborsRegressor<F>;
    type Error = FerroError;

    /// Fit the regressor by storing the training data and optionally
    /// building a KD-Tree spatial index.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of samples in
    /// `x` and `y` differ.
    /// Returns [`FerroError::InvalidParameter`] if `n_neighbors` is zero.
    ///
    /// `n_neighbors > n_samples` is NOT rejected at fit time (matching
    /// sklearn, which defers the `n_neighbors <= n_samples_fit` check to
    /// query time); it surfaces from `kneighbors`/`predict` instead.
    fn fit(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<FittedKNeighborsRegressor<F>, FerroError> {
        // sklearn validates CONSTRUCTOR PARAMS before the data: the
        // `@_fit_context` decorator runs `estimator._validate_params()`
        // (`sklearn/base.py:1466`) BEFORE the wrapped `fit` body's
        // `self._validate_data(X, y, ...)` finiteness check
        // (`:1472` → `NeighborsBase._fit`, `sklearn/neighbors/_base.py:475`). So
        // the `n_neighbors == 0` constructor-param check precedes the finiteness
        // guard; a doubly-degenerate input (invalid `n_neighbors` AND NaN/inf)
        // raises on `n_neighbors`, not the data. (#2273)
        if self.n_neighbors == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_neighbors".into(),
                reason: "must be at least 1".into(),
            });
        }

        // `_validate_data` keeps the default `force_all_finite=True`, so
        // `check_array` rejects any NaN or +/-inf in X OR (float) y with a
        // `ValueError` BEFORE the index is built, AFTER param validation (above)
        // and BEFORE the length check (below). `.iter().any(|v| !v.is_finite())`
        // rejects both NaN and Inf (bounds-safe, no panic, R-CODE-2). The
        // regressor `y` is `Array1<F>` (float targets), so both X and y are
        // guarded. (#2272, reordered #2273)
        if x.iter().any(|v| !v.is_finite()) {
            return Err(FerroError::InvalidParameter {
                name: "X".into(),
                reason: "Input X contains NaN or infinity.".into(),
            });
        }
        if y.iter().any(|v| !v.is_finite()) {
            return Err(FerroError::InvalidParameter {
                name: "y".into(),
                reason: "Input y contains NaN or infinity.".into(),
            });
        }

        let n_samples = x.nrows();

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        // sklearn does NOT validate n_neighbors vs n_samples at fit time
        // (`NeighborsBase._fit` has no such check); the
        // `n_neighbors <= n_samples_fit` test is deferred to query time in
        // `KNeighborsMixin.kneighbors` (sklearn/neighbors/_base.py:828-832).
        // ferrolearn mirrors this: `kneighbors_impl` already enforces
        // `n_neighbors > x_train.nrows()` at predict/kneighbors time, so no
        // fit-time guard is added here (#874).

        // Build spatial index.
        let spatial_index = build_spatial_index(self.algorithm, x);

        Ok(FittedKNeighborsRegressor {
            x_train: x.clone(),
            y_train: y.clone(),
            n_neighbors: self.n_neighbors,
            weights: self.weights,
            spatial_index,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedKNeighborsRegressor<F> {
    type Output = Array1<F>;
    type Error = FerroError;

    /// Predict target values for the given feature matrix.
    ///
    /// For each sample, finds the `k` nearest neighbors in the training
    /// data and returns the (weighted) mean of their target values.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features
    /// does not match the training data.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        let n_features = x.ncols();
        let train_features = self.x_train.ncols();

        if n_features != train_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![train_features],
                actual: vec![n_features],
                context: "number of features must match training data".into(),
            });
        }

        // Query-time finiteness guard (see classifier `predict`): sklearn
        // re-validates the query X (`_validate_data(..., reset=False)`,
        // `sklearn/neighbors/_base.py:824`) so a NaN/inf query raises before the
        // neighbor search rather than producing an OOB index panic. (#2274)
        if x.iter().any(|v| !v.is_finite()) {
            return Err(FerroError::InvalidParameter {
                name: "X".into(),
                reason: "Input X contains NaN or infinity.".into(),
            });
        }

        // Query-time `n_neighbors <= n_samples_fit` guard, deferred from fit
        // (sklearn/neighbors/_base.py:828-838 raises ValueError here for the
        // regressor too). Routes through the same `InsufficientSamples` error
        // as `kneighbors_impl` rather than clamping `k` via `find_neighbors`.
        let n_samples_fit = self.x_train.nrows();
        if self.n_neighbors > n_samples_fit {
            return Err(FerroError::InsufficientSamples {
                required: self.n_neighbors,
                actual: n_samples_fit,
                context: "n_neighbors exceeds number of training samples".into(),
            });
        }

        let n_samples = x.nrows();

        // Brute `ArgKmin` auto strategy resolved once for the whole call.
        let use_parallel_on_x = brute_use_parallel_on_x(n_samples_fit, n_samples);

        // Use a threshold to avoid Rayon overhead on small inputs.
        const PAR_THRESHOLD: usize = 256;

        let predictions_vec: Vec<F> = if n_samples >= PAR_THRESHOLD {
            (0..n_samples)
                .into_par_iter()
                .map(|i| {
                    let query: Vec<F> = (0..n_features).map(|j| x[[i, j]]).collect();
                    let neighbors = find_neighbors(
                        &self.x_train,
                        &query,
                        self.n_neighbors,
                        &self.spatial_index,
                        use_parallel_on_x,
                    );
                    self.weighted_mean(&neighbors)
                })
                .collect()
        } else {
            (0..n_samples)
                .map(|i| {
                    let query: Vec<F> = (0..n_features).map(|j| x[[i, j]]).collect();
                    let neighbors = find_neighbors(
                        &self.x_train,
                        &query,
                        self.n_neighbors,
                        &self.spatial_index,
                        use_parallel_on_x,
                    );
                    self.weighted_mean(&neighbors)
                })
                .collect()
        };

        Ok(Array1::from_vec(predictions_vec))
    }
}

impl<F: Float + Send + Sync + 'static> FittedKNeighborsRegressor<F> {
    /// Coefficient of determination R² on the given test data.
    ///
    /// Equivalent to sklearn's `RegressorMixin.score`. Returns
    /// `1 - SSres/SStot`, with the convention that constant-y returns 0
    /// when residuals are also zero, else `F::neg_infinity()` (sklearn
    /// returns 1.0 / -inf depending on residuals — we follow the latter
    /// for the genuine miss case).
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x.nrows() != y.len()` or
    /// the feature count does not match the training data.
    pub fn score(&self, x: &Array2<F>, y: &Array1<F>) -> Result<F, FerroError> {
        if x.nrows() != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows()],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }
        let preds = self.predict(x)?;
        Ok(r2_score(&preds, y))
    }

    /// Find the k nearest neighbors of each query sample in the training
    /// data. Mirrors sklearn `KNeighborsMixin.kneighbors`.
    ///
    /// Returns `(distances, indices)` where each is shape
    /// `(n_query_samples, n_neighbors_used)`.
    ///
    /// `n_neighbors` overrides the value set at construction; if `None`,
    /// uses `self.n_neighbors`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the query feature count
    /// does not match the training data, or
    /// [`FerroError::InsufficientSamples`] if `n_neighbors` exceeds the
    /// number of training samples.
    pub fn kneighbors(
        &self,
        x: &Array2<F>,
        n_neighbors: Option<usize>,
    ) -> Result<(Array2<F>, Array2<usize>), FerroError> {
        kneighbors_impl(
            &self.x_train,
            &self.spatial_index,
            x,
            n_neighbors.unwrap_or(self.n_neighbors),
        )
    }

    /// Number of training samples seen during `fit()`. Mirrors sklearn's
    /// `n_samples_fit_` attribute.
    #[must_use]
    pub fn n_samples_fit(&self) -> usize {
        self.x_train.nrows()
    }
}

/// R² coefficient of determination, used by both KNeighborsRegressor and
/// RadiusNeighborsRegressor.
pub(crate) fn r2_score<F: Float>(y_pred: &Array1<F>, y_true: &Array1<F>) -> F {
    let n = y_true.len();
    if n == 0 {
        return F::zero();
    }
    let mean = y_true.iter().copied().fold(F::zero(), |a, b| a + b) / F::from(n).unwrap();
    let mut ss_res = F::zero();
    let mut ss_tot = F::zero();
    for i in 0..n {
        let r = y_true[i] - y_pred[i];
        let t = y_true[i] - mean;
        ss_res = ss_res + r * r;
        ss_tot = ss_tot + t * t;
    }
    if ss_tot == F::zero() {
        // Constant target. sklearn returns 1.0 if perfect, else -inf.
        if ss_res == F::zero() {
            F::one()
        } else {
            F::neg_infinity()
        }
    } else {
        F::one() - ss_res / ss_tot
    }
}

/// Shared kneighbors implementation used by every fitted KNN-style
/// estimator. Validates feature count and `k`, then walks every query row
/// and returns aligned distance + index matrices.
pub(crate) fn kneighbors_impl<F: Float + Send + Sync + 'static>(
    x_train: &Array2<F>,
    spatial_index: &SpatialIndex,
    x: &Array2<F>,
    n_neighbors: usize,
) -> Result<(Array2<F>, Array2<usize>), FerroError> {
    let n_features = x.ncols();
    let train_features = x_train.ncols();
    if n_features != train_features {
        return Err(FerroError::ShapeMismatch {
            expected: vec![train_features],
            actual: vec![n_features],
            context: "number of features must match training data".into(),
        });
    }
    // Query-time finiteness guard: sklearn `KNeighborsMixin.kneighbors`
    // re-validates the query X with `self._validate_data(X, ..., reset=False)`
    // (`sklearn/neighbors/_base.py:824`, `force_all_finite=True`), so a NaN/inf
    // query raises `ValueError` before the neighbor search. Without it a NaN
    // would flow into the heap/kd_tree search and yield an OOB index. (#2274)
    if x.iter().any(|v| !v.is_finite()) {
        return Err(FerroError::InvalidParameter {
            name: "X".into(),
            reason: "Input X contains NaN or infinity.".into(),
        });
    }
    if n_neighbors == 0 {
        return Err(FerroError::InvalidParameter {
            name: "n_neighbors".into(),
            reason: "must be at least 1".into(),
        });
    }
    if n_neighbors > x_train.nrows() {
        return Err(FerroError::InsufficientSamples {
            required: n_neighbors,
            actual: x_train.nrows(),
            context: "n_neighbors exceeds number of training samples".into(),
        });
    }

    let n_queries = x.nrows();
    // Brute `ArgKmin` auto strategy resolved once for the whole call
    // (`_base.pyx.tp:183-196`): depends on the TOTAL query count.
    let use_parallel_on_x = brute_use_parallel_on_x(x_train.nrows(), n_queries);
    let mut distances = Array2::<F>::zeros((n_queries, n_neighbors));
    let mut indices = Array2::<usize>::zeros((n_queries, n_neighbors));

    for i in 0..n_queries {
        let query: Vec<F> = (0..n_features).map(|j| x[[i, j]]).collect();
        let neighbors = find_neighbors(
            x_train,
            &query,
            n_neighbors,
            spatial_index,
            use_parallel_on_x,
        );
        for (k, &(idx, dist)) in neighbors.iter().enumerate() {
            indices[[i, k]] = idx;
            distances[[i, k]] = dist;
        }
    }
    Ok((distances, indices))
}

impl<F: Float + Send + Sync + 'static> FittedKNeighborsClassifier<F> {
    /// Find the k nearest neighbors of each query sample in the training
    /// data. Mirrors sklearn `KNeighborsMixin.kneighbors`.
    ///
    /// Returns `(distances, indices)` of shape
    /// `(n_query_samples, n_neighbors_used)`.
    ///
    /// # Errors
    ///
    /// As the internal `kneighbors_impl`.
    pub fn kneighbors(
        &self,
        x: &Array2<F>,
        n_neighbors: Option<usize>,
    ) -> Result<(Array2<F>, Array2<usize>), FerroError> {
        kneighbors_impl(
            &self.x_train,
            &self.spatial_index,
            x,
            n_neighbors.unwrap_or(self.n_neighbors),
        )
    }

    /// Number of training samples seen during `fit()`. Mirrors sklearn's
    /// `n_samples_fit_` attribute.
    #[must_use]
    pub fn n_samples_fit(&self) -> usize {
        self.x_train.nrows()
    }
}

impl<F: Float + Send + Sync + 'static> FittedKNeighborsRegressor<F> {
    /// Compute the (possibly weighted) mean of neighbor targets.
    fn weighted_mean(&self, neighbors: &[(usize, F)]) -> F {
        let eps = F::from(1e-15).unwrap();

        match self.weights {
            Weights::Uniform => {
                let sum: F = neighbors
                    .iter()
                    .map(|&(idx, _)| self.y_train[idx])
                    .fold(F::zero(), |acc, v| acc + v);
                sum / F::from(neighbors.len()).unwrap()
            }
            Weights::Distance => {
                // Check if any neighbor has zero distance.
                let has_zero_dist = neighbors.iter().any(|&(_, d)| d < eps);

                if has_zero_dist {
                    // Average the targets of zero-distance neighbors.
                    let zero_neighbors: Vec<_> =
                        neighbors.iter().filter(|&&(_, d)| d < eps).collect();
                    let sum: F = zero_neighbors
                        .iter()
                        .map(|&&(idx, _)| self.y_train[idx])
                        .fold(F::zero(), |acc, v| acc + v);
                    sum / F::from(zero_neighbors.len()).unwrap()
                } else {
                    let mut weighted_sum = F::zero();
                    let mut weight_total = F::zero();
                    for &(idx, d) in neighbors {
                        let w = F::one() / d;
                        weighted_sum = weighted_sum + w * self.y_train[idx];
                        weight_total = weight_total + w;
                    }
                    weighted_sum / weight_total
                }
            }
        }
    }
}

// Pipeline integration.
impl<F: Float + Send + Sync + 'static> PipelineEstimator<F> for KNeighborsRegressor<F> {
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let fitted = self.fit(x, y)?;
        Ok(Box::new(FittedKNeighborsRegressorPipeline(fitted)))
    }
}

/// Wrapper for pipeline integration.
struct FittedKNeighborsRegressorPipeline<F: Float + Send + Sync + 'static>(
    FittedKNeighborsRegressor<F>,
);

unsafe impl<F: Float + Send + Sync + 'static> Send for FittedKNeighborsRegressorPipeline<F> {}
unsafe impl<F: Float + Send + Sync + 'static> Sync for FittedKNeighborsRegressorPipeline<F> {}

impl<F: Float + Send + Sync + 'static> FittedPipelineEstimator<F>
    for FittedKNeighborsRegressorPipeline<F>
{
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        self.0.predict(x)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    // -- Classifier Tests ---------------------------------------------------

    #[test]
    fn test_classifier_simple() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 5.0, 5.0, 5.5, 5.0, 5.0, 5.5],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let clf = KNeighborsClassifier::<f64>::new().with_n_neighbors(3);
        let fitted = clf.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        // All training points should be correctly classified with k=3.
        for i in 0..6 {
            assert_eq!(preds[i], y[i], "sample {i} misclassified");
        }
    }

    #[test]
    fn test_classifier_k1_memorizes() {
        // With k=1, the classifier should perfectly memorize training data.
        let x =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let y = array![0, 1, 2, 3];

        let clf = KNeighborsClassifier::<f64>::new().with_n_neighbors(1);
        let fitted = clf.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        for i in 0..4 {
            assert_eq!(preds[i], y[i], "k=1 should memorize training data");
        }
    }

    #[test]
    fn test_classifier_k_equals_n_predicts_mode() {
        // With k=n, every prediction should be the overall mode class.
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        // Mode is class 0 (appears 3 times).
        let y = array![0, 0, 0, 1, 1];

        let clf = KNeighborsClassifier::<f64>::new().with_n_neighbors(5);
        let fitted = clf.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        for i in 0..5 {
            assert_eq!(preds[i], 0, "k=n should predict the mode class");
        }
    }

    #[test]
    fn test_classifier_distance_weighting() {
        // Place test point at (0, 0). Nearest neighbor is class 0 at (0.1, 0),
        // while two class-1 points are far away at (10, 0) and (11, 0).
        let x = Array2::from_shape_vec((3, 1), vec![0.1, 10.0, 11.0]).unwrap();
        let y = array![0, 1, 1];

        let clf = KNeighborsClassifier::<f64>::new()
            .with_n_neighbors(3)
            .with_weights(Weights::Distance);
        let fitted = clf.fit(&x, &y).unwrap();

        // Query at origin: class 0 neighbor is much closer.
        let query = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();
        let preds = fitted.predict(&query).unwrap();
        assert_eq!(
            preds[0], 0,
            "distance weighting should favor closer neighbor"
        );
    }

    #[test]
    fn test_classifier_tied_votes() {
        // With uniform weights and k=2, both classes have 1 vote.
        // Tie-breaking should pick the smallest class label.
        let x = Array2::from_shape_vec((2, 1), vec![0.0, 1.0]).unwrap();
        let y = array![0, 1];

        let clf = KNeighborsClassifier::<f64>::new().with_n_neighbors(2);
        let fitted = clf.fit(&x, &y).unwrap();

        let query = Array2::from_shape_vec((1, 1), vec![0.5]).unwrap();
        let preds = fitted.predict(&query).unwrap();
        // Both are equidistant, both have 1 vote. Tie-break: smallest label wins.
        // However, tie-breaking depends on iteration order; we just check it doesn't panic.
        assert!(preds[0] == 0 || preds[0] == 1);
    }

    #[test]
    fn test_knn_tie_break_lowest_index_all_algorithms() -> Result<(), FerroError> {
        // k-th-boundary distance tie: query [0,0] coincides with idx0; idx1,
        // idx2, idx3 are all at distance 1 — a 3-way tie for the single k=2
        // slot. sklearn's KNeighborsClassifier (argpartition + argsort,
        // _base.py:738/:740-741) keeps the LOWEST tied index (idx1) and returns
        // the SAME result for every `algorithm`. ferrolearn must match: the
        // second neighbor is idx1 for brute/auto/kd_tree/ball_tree alike.
        let x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]];
        let y = array![5usize, 1, 0, 0];
        let xq = array![[0.0, 0.0]];

        for algo in [
            Algorithm::BruteForce,
            Algorithm::Auto,
            Algorithm::KdTree,
            Algorithm::BallTree,
        ] {
            let fitted = KNeighborsClassifier::<f64>::new()
                .with_n_neighbors(2)
                .with_algorithm(algo)
                .fit(&x, &y)?;
            let (_, indices) = fitted.kneighbors(&xq, None)?;
            // Nearest is idx0 (distance 0); the tie-break keeps idx1 (lowest
            // index among the three equidistant points), ordered second.
            assert_eq!(indices[[0, 0]], 0, "{algo:?}: nearest must be idx0");
            assert_eq!(
                indices[[0, 1]],
                1,
                "{algo:?}: k-th tie must break to the lowest index (idx1)"
            );
            // predict therefore returns class 1 (sklearn oracle), not class 0.
            let preds = fitted.predict(&xq)?;
            assert_eq!(
                preds[0], 1,
                "{algo:?}: predict must match sklearn (class 1)"
            );
        }
        Ok(())
    }

    #[test]
    #[allow(
        clippy::approx_constant,
        reason = "0.7071 is the exact literal of the #2141 sklearn-oracle fixture, \
                  not an approximation of FRAC_1_SQRT_2; changing it would change the oracle"
    )]
    fn test_neighbors_heap_matches_sklearn_kneighbors_order() -> Result<(), FerroError> {
        // Characterization (R-CHAR-3): the NeighborsHeap kernel reproduces
        // sklearn's exact kneighbors INDEX ORDER on exact-tie fixtures. Expected
        // orders are the LIVE sklearn 1.5.2 oracle (system python3), computed via
        //   from sklearn.neighbors import KNeighborsClassifier
        //   KNeighborsClassifier(n_neighbors=k, algorithm=A).fit(x,y).kneighbors([[0,0]])[1]
        // NOT copied from the ferrolearn side.
        //
        // Each fixture below is one where sklearn's order is well-defined (the
        // tie set resolves to a unique fill), so every algorithm string AND both
        // ArgKmin strategies agree — and the single-heap kernel matches.
        struct Case {
            x: Array2<f64>,
            k: usize,
            // sklearn kneighbors index order for query [0,0], all algorithms.
            order: Vec<usize>,
        }

        let cases = [
            // #2141 fixture: idx5/idx6 strictly closer (dist 0.99999041) than the
            // four unit-axis ties; k=4 keeps exactly one tied point (idx2).
            // sklearn (brute/kd_tree/ball_tree/auto): [0, 6, 5, 2].
            Case {
                x: array![
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [-1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, -1.0],
                    [0.7071, 0.7071],
                    [-0.7071, 0.7071],
                ],
                k: 4,
                order: vec![0, 6, 5, 2],
            },
            // #2139 fixture: idx0 at distance 0; idx1/idx2/idx3 tied at distance 1;
            // k=2 keeps idx0 then idx1. sklearn (all algorithms): [0, 1].
            Case {
                x: array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
                k: 2,
                order: vec![0, 1],
            },
            // Distinct distances (no ties): idx at 0, then 1, then 2 away.
            // sklearn order [0, 1, 2].
            Case {
                x: array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
                k: 3,
                order: vec![0, 1, 2],
            },
        ];

        let y_dummy = |n: usize| Array1::from_iter(0..n);
        let xq = array![[0.0, 0.0]];

        for case in cases {
            let n = case.x.nrows();
            for algo in [
                Algorithm::BruteForce,
                Algorithm::Auto,
                Algorithm::KdTree,
                Algorithm::BallTree,
            ] {
                let fitted = KNeighborsClassifier::<f64>::new()
                    .with_n_neighbors(case.k)
                    .with_algorithm(algo)
                    .fit(&case.x, &y_dummy(n))?;
                let (_d, indices) = fitted.kneighbors(&xq, Some(case.k))?;
                let got: Vec<usize> = indices.row(0).to_vec();
                assert_eq!(
                    got, case.order,
                    "{algo:?}: NeighborsHeap order must equal sklearn kneighbors \
                     order {:?}",
                    case.order
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_classifier_brute_force_algorithm() {
        let x =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let y = array![0, 1, 0, 1];

        let clf = KNeighborsClassifier::<f64>::new()
            .with_n_neighbors(1)
            .with_algorithm(Algorithm::BruteForce);
        let fitted = clf.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        for i in 0..4 {
            assert_eq!(preds[i], y[i]);
        }
    }

    #[test]
    fn test_classifier_kdtree_algorithm() {
        let x =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let y = array![0, 1, 0, 1];

        let clf = KNeighborsClassifier::<f64>::new()
            .with_n_neighbors(1)
            .with_algorithm(Algorithm::KdTree);
        let fitted = clf.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        for i in 0..4 {
            assert_eq!(preds[i], y[i]);
        }
    }

    #[test]
    fn test_classifier_shape_mismatch_fit() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![0, 1]; // Wrong length.

        let clf = KNeighborsClassifier::<f64>::new();
        assert!(clf.fit(&x, &y).is_err());
    }

    #[test]
    fn test_classifier_shape_mismatch_predict() {
        let x =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let clf = KNeighborsClassifier::<f64>::new().with_n_neighbors(1);
        let fitted = clf.fit(&x, &y).unwrap();

        // Wrong number of features.
        let x_bad = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_classifier_invalid_k() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![0, 0, 1];

        let clf = KNeighborsClassifier::<f64>::new().with_n_neighbors(0);
        assert!(clf.fit(&x, &y).is_err());
    }

    #[test]
    fn test_classifier_fit_succeeds_when_k_gt_n() {
        // sklearn does NOT validate n_neighbors vs n_samples at fit time;
        // KNeighborsClassifier(n_neighbors=5).fit(X_2rows, y) SUCCEEDS and the
        // `n_neighbors <= n_samples_fit` check is deferred to query time
        // (sklearn/neighbors/_base.py:828-832). ferrolearn mirrors this.
        let x = array![[1.0], [2.0]];
        let y = array![0, 1];

        let clf = KNeighborsClassifier::<f64>::new().with_n_neighbors(5);
        let fitted = clf.fit(&x, &y);
        assert!(
            fitted.is_ok(),
            "fit with k=5 > n=2 must succeed (sklearn defers the check to query time)"
        );

        if let Ok(fitted) = fitted {
            // `kneighbors` carries the explicit `k > n_train` guard, so the
            // query-time error surfaces there (sklearn raises ValueError).
            assert!(
                fitted.kneighbors(&x, None).is_err(),
                "kneighbors with k=5 > n_train=2 must error at query time"
            );

            // `predict`/`predict_proba` carry the SAME query-time guard
            // (sklearn raises `ValueError("Expected n_neighbors <=
            // n_samples_fit, ...")` at predict time, `_base.py:828-838`); they
            // must NOT clamp k to n_train and silently return a prediction
            // (#2140).
            assert!(
                fitted.predict(&x).is_err(),
                "predict with k=5 > n_train=2 must error at query time (sklearn parity)"
            );
            assert!(
                fitted.predict_proba(&x).is_err(),
                "predict_proba with k=5 > n_train=2 must error at query time (sklearn parity)"
            );
        }
    }

    #[test]
    fn test_classifier_has_classes() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0, 1, 2, 1];

        let clf = KNeighborsClassifier::<f64>::new().with_n_neighbors(1);
        let fitted = clf.fit(&x, &y).unwrap();

        assert_eq!(fitted.classes(), &[0, 1, 2]);
        assert_eq!(fitted.n_classes(), 3);
    }

    #[test]
    fn test_classifier_single_neighbor() {
        let x = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
        let y = array![42];

        let clf = KNeighborsClassifier::<f64>::new().with_n_neighbors(1);
        let fitted = clf.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds[0], 42);
    }

    #[test]
    fn test_classifier_pipeline_integration() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 5.0, 5.0, 5.5, 5.0, 5.0, 5.5],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let clf = KNeighborsClassifier::<f64>::new().with_n_neighbors(3);
        let fitted = clf.fit_pipeline(&x, &y).unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_classifier_f32_support() {
        let x = Array2::from_shape_vec((4, 2), vec![0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
            .unwrap();
        let y = array![0, 1, 0, 1];

        let clf = KNeighborsClassifier::<f32>::new().with_n_neighbors(1);
        let fitted = clf.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    // -- Regressor Tests ----------------------------------------------------

    #[test]
    fn test_regressor_simple() {
        // y = 2*x, k=1 should memorize.
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let reg = KNeighborsRegressor::<f64>::new().with_n_neighbors(1);
        let fitted = reg.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        for i in 0..5 {
            assert_relative_eq!(preds[i], y[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_regressor_mean_of_neighbors() {
        // k=3, query at center should predict mean of 3 nearest.
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 100.0]).unwrap();
        let y = array![0.0, 10.0, 20.0, 30.0, 1000.0];

        let reg = KNeighborsRegressor::<f64>::new().with_n_neighbors(3);
        let fitted = reg.fit(&x, &y).unwrap();

        // Query at 1.0: nearest are indices 0, 1, 2 with targets 0, 10, 20.
        let query = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        let preds = fitted.predict(&query).unwrap();
        assert_relative_eq!(preds[0], 10.0, epsilon = 1e-10);
    }

    #[test]
    fn test_regressor_distance_weighting() {
        // Two points: (0, target=0) and (10, target=100).
        // Query at 1.0: closer to (0), so distance-weighted should bias toward 0.
        let x = Array2::from_shape_vec((2, 1), vec![0.0, 10.0]).unwrap();
        let y = array![0.0, 100.0];

        let reg = KNeighborsRegressor::<f64>::new()
            .with_n_neighbors(2)
            .with_weights(Weights::Distance);
        let fitted = reg.fit(&x, &y).unwrap();

        let query = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        let preds = fitted.predict(&query).unwrap();

        // Weight for (0): 1/1 = 1.0, weight for (10): 1/9 ~ 0.111.
        // Weighted mean: (1.0*0.0 + 0.111*100.0) / (1.0 + 0.111) = 11.11 / 1.111 ~ 10.0
        let expected = (1.0 * 0.0 + (1.0 / 9.0) * 100.0) / (1.0 + 1.0 / 9.0);
        assert_relative_eq!(preds[0], expected, epsilon = 1e-6);
    }

    #[test]
    fn test_regressor_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![1.0, 2.0]; // Wrong length.

        let reg = KNeighborsRegressor::<f64>::new();
        assert!(reg.fit(&x, &y).is_err());
    }

    #[test]
    fn test_regressor_invalid_k() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];

        let reg = KNeighborsRegressor::<f64>::new().with_n_neighbors(0);
        assert!(reg.fit(&x, &y).is_err());
    }

    #[test]
    fn test_regressor_pipeline_integration() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let reg = KNeighborsRegressor::<f64>::new().with_n_neighbors(3);
        let fitted = reg.fit_pipeline(&x, &y).unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 5);
    }

    #[test]
    fn test_regressor_f32_support() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0f32, 2.0, 3.0]).unwrap();
        let y = Array1::from_vec(vec![2.0f32, 4.0, 6.0]);

        let reg = KNeighborsRegressor::<f32>::new().with_n_neighbors(1);
        let fitted = reg.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 3);
    }

    #[test]
    fn test_high_dimensional_falls_back_to_brute_force() {
        // With d > 20 and Algorithm::Auto, should use brute force (no KD-Tree).
        let n_features = 25;
        let n_samples = 10;
        let data: Vec<f64> = (0..n_samples * n_features).map(|i| i as f64).collect();
        let x = Array2::from_shape_vec((n_samples, n_features), data).unwrap();
        let y = Array1::from_vec(vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1]);

        let clf = KNeighborsClassifier::<f64>::new()
            .with_n_neighbors(3)
            .with_algorithm(Algorithm::Auto);
        let fitted = clf.fit(&x, &y).unwrap();

        // With d > 20 and Auto, should use BallTree (not brute force).
        assert!(matches!(fitted.spatial_index, SpatialIndex::BallTree(_)));

        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), n_samples);
    }

    #[test]
    fn test_classifier_default() {
        let clf = KNeighborsClassifier::<f64>::default();
        assert_eq!(clf.n_neighbors, 5);
        assert_eq!(clf.algorithm, Algorithm::Auto);
        assert_eq!(clf.weights, Weights::Uniform);
    }

    #[test]
    fn test_regressor_default() {
        let reg = KNeighborsRegressor::<f64>::default();
        assert_eq!(reg.n_neighbors, 5);
        assert_eq!(reg.algorithm, Algorithm::Auto);
        assert_eq!(reg.weights, Weights::Uniform);
    }

    #[test]
    fn test_classifier_new_data_prediction() {
        // Train on two clear clusters and predict new points.
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let clf = KNeighborsClassifier::<f64>::new().with_n_neighbors(3);
        let fitted = clf.fit(&x, &y).unwrap();

        // New test points.
        let x_test = Array2::from_shape_vec((2, 2), vec![0.05, 0.05, 4.9, 4.9]).unwrap();
        let preds = fitted.predict(&x_test).unwrap();
        assert_eq!(preds[0], 0);
        assert_eq!(preds[1], 1);
    }

    #[test]
    fn test_regressor_exact_match_distance_weighting() {
        // When query exactly matches a training point with distance weighting.
        let x = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
        let y = array![10.0, 20.0, 30.0];

        let reg = KNeighborsRegressor::<f64>::new()
            .with_n_neighbors(3)
            .with_weights(Weights::Distance);
        let fitted = reg.fit(&x, &y).unwrap();

        // Query exactly at x=1.0.
        let query = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        let preds = fitted.predict(&query).unwrap();
        // Should return 20.0 (exact match takes all weight).
        assert_relative_eq!(preds[0], 20.0, epsilon = 1e-10);
    }
}
