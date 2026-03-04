# Feature: Fix Benchmark Regressions Against scikit-learn

## Summary

Ferrolearn beats sklearn in 74/84 benchmarks (geometric mean 9.4x faster), but loses badly in 3 areas: LinearRegression fit (up to 237x slower), KNeighborsClassifier predict on high-dimensional data (68x slower), and KMeans predict/fit on small inputs (up to 10x slower). This design targets all three regressions with concrete, minimal changes to existing code.

## Benchmark Evidence

Current regressions from `ferrolearn-bench` vs `scripts/benchmark_sklearn.py`:

| Benchmark | Rust (ms) | sklearn (ms) | Ratio |
|-----------|-----------|--------------|-------|
| LinearRegression/fit/tiny_50x5 | 9.00 | 0.42 | 21.7x slower |
| LinearRegression/fit/small_1Kx10 | 130.90 | 0.55 | 237x slower |
| LinearRegression/fit/medium_10Kx100 | 1458.25 | 440.44 | 3.3x slower |
| KNN/predict/small_1Kx10 | 34.86 | 5.87 | 5.9x slower |
| KNN/predict/medium_10Kx100 | 8986.96 | 131.97 | 68x slower |
| KMeans/predict/tiny_50x5 | 0.85 | 0.08 | 10.3x slower |
| KMeans/predict/small_1Kx10 | 2.01 | 0.22 | 9.2x slower |
| KMeans/fit/tiny_50x5 | 7.80 | 2.13 | 3.7x slower |
| GaussianNB/predict/medium_10Kx100 | 6.81 | 4.13 | 1.6x slower |
| StandardScaler/transform/medium_10Kx100 | 2.21 | 1.63 | 1.4x slower |

## Requirements

- REQ-1: LinearRegression fit must be within 2x of sklearn at all three dataset sizes (tiny, small, medium)
- REQ-2: KNeighborsClassifier predict must be within 3x of sklearn on datasets with >20 features
- REQ-3: KMeans predict must not use Rayon for datasets under 1000 samples, eliminating thread-pool overhead
- REQ-4: KMeans fit must reuse allocations across Lloyd iterations instead of reallocating per iteration
- REQ-5: All existing tests must continue to pass with identical numerical results (within existing tolerances)
- REQ-6: No new public API changes; all fixes are internal implementation details
- REQ-7: GaussianNB predict and StandardScaler transform medium-scale regressions should be investigated and fixed if a clear low-hanging optimization exists

## Acceptance Criteria

- [ ] AC-1: `cargo bench --bench regressors -- LinearRegression/fit` shows median times within 2x of sklearn JSON baseline at all sizes
- [ ] AC-2: `cargo bench --bench classifiers -- KNeighborsClassifier/predict` shows median times within 3x of sklearn at all sizes including medium_10Kx100
- [ ] AC-3: `cargo bench --bench clusterers -- KMeans/predict/tiny` shows median time under 0.2ms (currently 0.85ms)
- [ ] AC-4: `cargo bench --bench clusterers -- KMeans/fit/tiny` shows median time under 4ms (currently 7.8ms)
- [ ] AC-5: `cargo test --workspace` passes with zero failures
- [ ] AC-6: Benchmark improvements verified by running `scripts/benchmark_sklearn.py` and comparing Rust criterion output side-by-side

## Architecture

### Fix 1: LinearRegression Fit — Switch from QR to Cholesky Normal Equations

**Files:** `ferrolearn-linear/src/linear_regression.rs`, `ferrolearn-linear/src/linalg.rs`

**Root cause analysis:**

The current `LinearRegression::fit()` pipeline has three compounding performance problems:

1. **Matrix augmentation** (line 125-130 of `linear_regression.rs`): When `fit_intercept=true`, it allocates an entirely new `(n, p+1)` matrix by calling `ndarray::concatenate`. For 10K×100, that's an 8MB memcpy before any math happens.

2. **Element-wise ndarray→faer conversion** (line 16 of `linalg.rs`): `faer::Mat::from_fn(nrows, ncols, |i, j| a[[i, j]])` does random-access indexing across a potentially non-contiguous ndarray. This is cache-hostile and O(n*p) with bad constants.

3. **QR decomposition overkill**: Full QR decomposition via faer is O(np^2) with good numerical stability, but for well-conditioned OLS problems it's unnecessarily expensive. Ridge uses Cholesky on the normal equations and is 3-340x faster than LinearRegression at the same data sizes. sklearn's `LinearRegression` also uses LAPACK's `dgelsd` (SVD-based) but with MKL-optimized BLAS underneath.

**Proposed fix — adopt Ridge's strategy:**

Use the centering trick (like `ferrolearn-linear/src/ridge.rs` lines 140-160) to avoid matrix augmentation, then solve via Cholesky on the normal equations using the existing `Backend` trait:

```rust
// In linear_regression.rs fit():
if self.fit_intercept {
    // Center X and y (like Ridge does) — no matrix augmentation needed
    let x_mean = x.mean_axis(Axis(0)).unwrap();
    let y_mean = y.mean().unwrap();
    let x_centered = x - &x_mean;
    let y_centered = y - y_mean;

    // Solve centered system via Cholesky (same as Ridge with alpha=0)
    // Use a tiny regularization (1e-12) to guarantee positive-definiteness
    let w = linalg::solve_normal_equations(&x_centered, &y_centered)?;

    // Recover intercept: b = y_mean - x_mean . w
    let intercept = y_mean - x_mean.dot(&w);
    Ok(FittedLinearRegression { coefficients: w, intercept })
} else {
    let w = linalg::solve_normal_equations(x, y)?;
    Ok(FittedLinearRegression { coefficients: w, intercept: F::zero() })
}
```

**Why this works:**

- `x.mean_axis()` and centering are O(np) — trivial compared to decomposition
- `X^T X` is a p×p matrix (10×10 or 100×100) — Cholesky on this is near-instant
- No ndarray→faer conversion needed; the existing pure-ndarray `cholesky_solve` handles this
- Ridge already proves this strategy works and is fast (0.0008ms at tiny, 4.65ms at medium)

**Why not keep QR as a fallback?**

QR is more numerically stable for rank-deficient matrices. The fix should try Cholesky first and fall back to QR (via faer) only if Cholesky fails. This matches sklearn's approach (try fast path, fall back to SVD).

**Estimated speedup:** 100-200x at small sizes (dominated by overhead elimination), 3-5x at medium (Cholesky vs QR).

**Make `solve_normal_equations` public within the crate:**

Currently `solve_normal_equations` is private. Change it to `pub(crate)` so `linear_regression.rs` can call it directly. No changes to `linalg.rs` logic needed.

### Fix 2: KNeighborsClassifier Predict — Ball Tree for High Dimensions

**Files:** `ferrolearn-neighbors/src/kdtree.rs`, `ferrolearn-neighbors/src/knn.rs`

**Root cause analysis:**

The KD-tree auto-selection threshold is `n_features <= 20` (`knn.rs` line 111-116). For the benchmark's medium dataset (10K×100 features), this correctly falls back to brute force. But brute force KNN is O(n*k*d) per query point, totaling O(n^2 * d) for predicting on the training set. At 10K×100, that's 10 billion float operations.

sklearn uses a **ball tree** for moderate-to-high dimensions, which partitions data by hypersphere enclosure rather than axis-aligned splits. Ball trees degrade gracefully with dimensionality (unlike KD-trees which become O(n) at ~20+ dims).

**Proposed fix — implement BallTree:**

Add a `BallTree` struct to `ferrolearn-neighbors/src/balltree.rs`:

```rust
pub struct BallTree {
    nodes: Vec<BallNode>,  // Flat array layout (cache-friendly)
    data: Vec<f64>,        // Flattened training data (n_samples * n_features)
    n_features: usize,
    indices: Vec<usize>,   // Original indices for leaf nodes
}

struct BallNode {
    center: usize,    // Index into data array for center point
    radius: f64,      // Bounding radius
    left: u32,        // Index of left child in nodes array
    right: u32,       // Index of right child in nodes array
    start: u32,       // Start index in indices array (leaf)
    end: u32,         // End index in indices array (leaf)
}
```

**Build algorithm** (O(n log n)):
1. Pick the dimension of greatest spread
2. Project all points onto that dimension
3. Split at the median
4. Compute bounding ball (center = centroid, radius = max distance to any point)
5. Recurse on left and right halves

**Query algorithm** (O(n^(1-1/d) log n) amortized):
1. Priority queue search: expand closest node first
2. Prune if `distance_to_ball_center - radius > current_kth_distance`
3. Triangle inequality pruning on the bounding ball

**Update auto-selection in `knn.rs`:**

```rust
fn should_use_kdtree(algorithm: Algorithm, n_features: usize) -> bool {
    match algorithm {
        Algorithm::Auto => n_features <= 15,  // tighten KD-tree threshold
        Algorithm::BruteForce => false,
        Algorithm::KdTree => true,
        Algorithm::BallTree => false,  // handled separately
    }
}

fn should_use_balltree(algorithm: Algorithm, n_features: usize) -> bool {
    match algorithm {
        Algorithm::Auto => n_features > 15,
        Algorithm::BallTree => true,
        _ => false,
    }
}
```

Add `Algorithm::BallTree` variant to the existing enum.

**Estimated speedup:** 10-50x for 100-dimensional data depending on data distribution. sklearn's ball tree achieves ~130ms on 10K×100 vs our 9000ms brute force.

### Fix 3: KMeans — Conditional Parallelism and Allocation Reuse

**Files:** `ferrolearn-cluster/src/kmeans.rs`

**Root cause analysis:**

Two problems compound on small inputs:

1. **Rayon overhead** (`assign_clusters` line 227): `(0..n_samples).into_par_iter()` spawns thread-pool work for 50 samples. Rayon's work-stealing has ~10-50μs overhead per dispatch, which dominates when the actual work is ~1μs.

2. **Per-iteration allocation** (`assign_clusters` line 246, `recompute_centroids` line 266-267): Every Lloyd iteration allocates a new `Array1<usize>` for labels and `Array2<f64>` for centers, plus a `Vec<(usize, F)>` intermediate. For n_init=3 with ~10 iterations each, that's ~30 unnecessary allocations.

**Proposed fix A — conditional parallelism:**

```rust
const PARALLEL_THRESHOLD: usize = 512;

fn assign_clusters<F: Float + Send + Sync>(
    x: &Array2<F>,
    centers: &Array2<F>,
) -> (Array1<usize>, F) {
    let n_samples = x.nrows();

    if n_samples >= PARALLEL_THRESHOLD {
        assign_clusters_parallel(x, centers)
    } else {
        assign_clusters_serial(x, centers)
    }
}
```

The serial path writes directly into pre-allocated `Array1<usize>` + accumulates inertia in a scalar, avoiding the intermediate `Vec<(usize, F)>`.

**Proposed fix B — allocation reuse in fit loop:**

```rust
// Pre-allocate ONCE before the n_init loop
let mut labels = Array1::zeros(n_samples);
let mut new_centers = Array2::zeros((self.n_clusters, n_features));
let mut counts = vec![F::zero(); self.n_clusters];

for run in 0..self.n_init {
    // ... kmeans++ init ...
    for iter in 0..self.max_iter {
        // Reuse `labels` buffer
        let inertia = assign_clusters_into(&mut labels, x, &centers);

        // Reuse `new_centers` and `counts` buffers
        new_centers.fill(F::zero());
        counts.fill(F::zero());
        recompute_centroids_into(&mut new_centers, &mut counts, x, &labels, ...);
    }
}
```

**Proposed fix C — fuse assign + inertia in parallel path:**

Replace `collect::<Vec<(usize, F)>>` followed by sequential iteration with a parallel fold that directly populates the labels array:

```rust
fn assign_clusters_parallel<F: Float + Send + Sync>(
    x: &Array2<F>,
    centers: &Array2<F>,
    labels: &mut Array1<usize>,
) -> F {
    let n_samples = x.nrows();
    let labels_slice = labels.as_slice_mut().unwrap();

    // Use par_chunks for cache-friendly access
    let chunk_size = (n_samples / rayon::current_num_threads()).max(64);

    labels_slice
        .par_chunks_mut(chunk_size)
        .enumerate()
        .map(|(chunk_idx, chunk)| {
            let start = chunk_idx * chunk_size;
            let mut local_inertia = F::zero();
            for (local_i, label) in chunk.iter_mut().enumerate() {
                let i = start + local_i;
                // ... find closest center, write label, accumulate inertia ...
            }
            local_inertia
        })
        .sum()
}
```

**Estimated speedup:** 3-10x on tiny/small inputs (overhead elimination), neutral on medium (already parallel).

### Fix 4: GaussianNB Predict — Investigate and Micro-optimize (Low Priority)

**Files:** `ferrolearn-bayes/src/gaussian_nb.rs`

The regression is small (1.6x) and only at medium scale. Likely cause is that sklearn's GaussianNB predict uses vectorized numpy for the log-likelihood computation across all classes simultaneously, while our implementation may loop over classes sequentially. Investigation needed to confirm.

**Approach:** Profile with `cargo flamegraph`, identify if the bottleneck is log/exp computation or memory layout. If it's a simple vectorization opportunity (e.g., computing log-likelihoods as a matrix multiply), fix it. If it requires major restructuring, defer.

### Fix 5: StandardScaler Transform — Investigate (Low Priority)

**Files:** `ferrolearn-preprocess/src/standard_scaler.rs`

The regression is only 1.4x at medium scale. Likely cause: ndarray's element-wise `(x - mean) / std` generates intermediate arrays, while numpy fuses these operations. This may require using `Zip::from(x).and(mean).and(std).for_each()` to avoid intermediates.

**Approach:** Check if the current implementation allocates intermediate arrays. If so, switch to in-place or fused iteration. If already fused, the 1.4x gap is acceptable (numpy's C loops vs Rust ndarray iteration).

## Open Questions

None — all three major regressions have clear root causes and proven fix strategies (Ridge already demonstrates the Cholesky approach works, sklearn documents ball-tree behavior, and Rayon's overhead on small workloads is well-known).

## Out of Scope

- Adding a BLAS/MKL backend: would help LinearRegression at very large scale but is a separate, larger initiative (see design doc Section 11 of `rust-ml-design.md`)
- Parallelizing LinearRegression fit itself: not worth it since the bottleneck is the decomposition, not data parallelism
- Optimizing KNN for the Minkowski metric family: current scope is Euclidean only
- GPU acceleration for any of these algorithms
- Changing public API signatures or adding new public types (except `Algorithm::BallTree` enum variant, which is additive)
