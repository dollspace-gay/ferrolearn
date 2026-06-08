//! K-bins discretizer: bin continuous features into discrete intervals.
//!
//! [`KBinsDiscretizer`] transforms continuous features into discrete bins.
//! Each feature is independently binned according to one of the following
//! strategies:
//!
//! - [`BinStrategy::Uniform`] — equal-width bins.
//! - [`BinStrategy::Quantile`] — bins with equal numbers of samples.
//! - [`BinStrategy::KMeans`] — bins based on 1D k-means clustering.
//!
//! The output can be ordinal-encoded (integers 0..k-1) or one-hot encoded.
//!
//! ## REQ status
//!
//! Translation target: scikit-learn 1.5.2 `class KBinsDiscretizer`
//! (`sklearn/preprocessing/_discretization.py:184`). Tracking: #1375. Each REQ
//! is BINARY — SHIPPED (impl + non-test consumer + tests + green verification)
//! or NOT-STARTED (with a concrete open blocker).
//!
//! | REQ | Scope | Status | Evidence / Blocker |
//! |-----|-------|--------|--------------------|
//! | REQ-1 | Uniform + Quantile bin EDGES + ordinal/onehot transform VALUES (non-degenerate features) | SHIPPED | [`KBinsDiscretizer`] `fit` — Uniform=`np.linspace` (`_discretization.py:271`), Quantile=`np.percentile` (`:276`); `assign_bin` ≡ `searchsorted(edges[1:-1], side="right")` (`:377`); oracle value tests in `tests/divergence_kbins_discretizer.rs`. Consumer: re-export `lib.rs:151` |
//! | REQ-2 | KMeans bin edges/transform via faithful sklearn `KMeans` Lloyd | SHIPPED | `kmeans_1d` replicates sklearn `KMeans` Lloyd incl. mean-centering (`_kmeans.py:1486-1546`), `\|\|C\|\|²-2xC` assignment + lowest-index tie-break (`_k_means_lloyd.pyx:196-213`), empty-cluster RELOCATION (`_k_means_common.pyx:_relocate_empty_clusters_dense`), var-scaled `tol=mean(var)*1e-4` (`_tolerance`), strict/center-shift convergence, max_iter=300, deterministic uniform-center init (`_discretization.py:289-300`); matches sklearn bit-for-bit on well-separated + moderately-separated + empty-init-cluster data (km1 #2321, km2 #2322, 3 green oracle fixtures). RESIDUAL: ~0.1% of well-spread continuous data converges to a different valid Lloyd local optimum (BLAS-gemm vs scalar float tie-break) — honestly pinned `divergence_km3_blas_gemm_local_optimum` (#2321 follow-up) |
//! | REQ-3 | Error/parameter contracts (n_samples<2, n_bins<2, transform ncols, unfitted) | SHIPPED (scoped) | [`KBinsDiscretizer::fit`]/[`FittedKBinsDiscretizer`] `transform`; in-module + divergence error tests |
//! | REQ-4 | Constant feature → bin 0 + per-feature `n_bins_=1` (`col_min==col_max`) | SHIPPED | `fit` sets `bin_edges=[-inf,+inf]` + `n_bins_per_feature[j]=1`; `assign_bin` → bin 0 (mirrors `_discretization.py:262-268`); 3 oracle tests — was DIV-1 #1376, fixed |
//! | REQ-5 | Small-bin removal (quantile/kmeans near-duplicate edge collapse → per-feature `n_bins_`) + onehot variable width | SHIPPED | `fit` collapse `gap > 1e-8` (mirrors `ediff1d > 1e-8` `:302-312`); `transform` onehot width = `sum(n_bins_per_feature)` cumulative offsets; oracle tests (quantile collapse, onehot variable width, threshold boundary) — was DIV-2 #1377, fixed |
//! | REQ-6 | `subsample` (default 200000) + `random_state` resample for quantile/kmeans | NOT-STARTED | sklearn `_discretization.py:242-249` — blocker #1379 |
//! | REQ-7 | `n_bins` as per-feature array + `_validate_n_bins` | NOT-STARTED | scalar only; sklearn `_discretization.py:329-352` — blocker #1380 |
//! | REQ-8 | encode='onehot' SPARSE default + sklearn ctor defaults (encode=onehot, strategy=quantile) | NOT-STARTED | dense only, defaults Ordinal/Uniform; sklearn `_discretization.py:185,320` — blocker #1381 |
//! | REQ-9 | `dtype` param + `sample_weight` (weighted percentile/kmeans) | NOT-STARTED | sklearn `_discretization.py:228,235,295` — blocker #1382 |
//! | REQ-10 | `inverse_transform` | NOT-STARTED | sklearn `_discretization.py:393` — blocker #1383 |
//! | REQ-11 | `get_feature_names_out` + `bin_edges_`/`n_bins_` attr names + `PipelineTransformer` impl | NOT-STARTED | absent — blocker #1384 |
//! | REQ-12 | PyO3 binding | NOT-STARTED | no `ferrolearn-python` registration — blocker #1385 |
//! | REQ-13 | ferray substrate | NOT-STARTED | dense `Array2` + `num_traits::Float` only — blocker #1386 |
//! | REQ-14 | KMeans empty-cluster relocation (degenerate/duplicate-heavy data) | SHIPPED | `kmeans_1d` now RELOCATES empty clusters (translates `_relocate_empty_clusters_dense`, `_k_means_common.pyx:170-214`): farthest-point reassignment + heaviest-cluster fallback in `_average_centers`; km2 (#2322) + `green_kmeans_empty_cluster_relocation_k3`. Residual degenerate carve-out (more clusters than distinct samples / multi-empty argpartition tie order) folded into REQ-2's pinned BLAS-gemm residual — was carve-out #1378 |

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, FitTransform, Transform};
use ndarray::Array2;
use num_traits::Float;

// ---------------------------------------------------------------------------
// BinStrategy
// ---------------------------------------------------------------------------

/// Strategy for computing bin edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinStrategy {
    /// Equal-width bins.
    Uniform,
    /// Equal-frequency bins (quantile-based).
    Quantile,
    /// Bins based on 1D k-means clustering.
    KMeans,
}

/// Encoding method for the output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinEncoding {
    /// Ordinal encoding: each value is replaced by its bin index (0..n_bins-1).
    Ordinal,
    /// One-hot encoding: each bin becomes a separate binary column.
    OneHot,
}

// ---------------------------------------------------------------------------
// KBinsDiscretizer (unfitted)
// ---------------------------------------------------------------------------

/// An unfitted K-bins discretizer.
///
/// Calling [`Fit::fit`] computes the bin edges for each feature and returns a
/// [`FittedKBinsDiscretizer`].
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::kbins_discretizer::{KBinsDiscretizer, BinStrategy, BinEncoding};
/// use ferrolearn_core::traits::{Fit, Transform};
/// use ndarray::array;
///
/// let disc = KBinsDiscretizer::<f64>::new(3, BinEncoding::Ordinal, BinStrategy::Uniform);
/// let x = array![[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]];
/// let fitted = disc.fit(&x, &()).unwrap();
/// let out = fitted.transform(&x).unwrap();
/// // Values should be in {0.0, 1.0, 2.0}
/// for v in out.iter() {
///     assert!(*v >= 0.0 && *v < 3.0);
/// }
/// ```
#[must_use]
#[derive(Debug, Clone)]
pub struct KBinsDiscretizer<F> {
    /// Number of bins.
    n_bins: usize,
    /// Encoding method.
    encode: BinEncoding,
    /// Binning strategy.
    strategy: BinStrategy,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> KBinsDiscretizer<F> {
    /// Create a new `KBinsDiscretizer`.
    pub fn new(n_bins: usize, encode: BinEncoding, strategy: BinStrategy) -> Self {
        Self {
            n_bins,
            encode,
            strategy,
            _marker: std::marker::PhantomData,
        }
    }

    /// Return the number of bins.
    #[must_use]
    pub fn n_bins(&self) -> usize {
        self.n_bins
    }

    /// Return the encoding method.
    #[must_use]
    pub fn encode(&self) -> BinEncoding {
        self.encode
    }

    /// Return the binning strategy.
    #[must_use]
    pub fn strategy(&self) -> BinStrategy {
        self.strategy
    }
}

impl<F: Float + Send + Sync + 'static> Default for KBinsDiscretizer<F> {
    fn default() -> Self {
        Self::new(5, BinEncoding::Ordinal, BinStrategy::Uniform)
    }
}

// ---------------------------------------------------------------------------
// FittedKBinsDiscretizer
// ---------------------------------------------------------------------------

/// A fitted K-bins discretizer holding per-feature bin edges.
///
/// Created by calling [`Fit::fit`] on a [`KBinsDiscretizer`].
#[derive(Debug, Clone)]
pub struct FittedKBinsDiscretizer<F> {
    /// Bin edges per feature. `bin_edges[j]` has `n_bins_per_feature[j] + 1`
    /// edges (which may be fewer than `n_bins + 1` for constant or collapsed
    /// features).
    bin_edges: Vec<Vec<F>>,
    /// Per-feature bin count (mirrors sklearn `n_bins_`). A constant feature
    /// collapses to 1 bin; quantile/kmeans features may shrink when
    /// near-duplicate edges are removed.
    n_bins_per_feature: Vec<usize>,
    /// Requested number of bins (the global `n_bins` argument).
    n_bins: usize,
    /// Encoding method.
    encode: BinEncoding,
}

impl<F: Float + Send + Sync + 'static> FittedKBinsDiscretizer<F> {
    /// Return the bin edges per feature.
    #[must_use]
    pub fn bin_edges(&self) -> &[Vec<F>] {
        &self.bin_edges
    }

    /// Return the per-feature bin count (sklearn `n_bins_`).
    #[must_use]
    pub fn n_bins_per_feature(&self) -> &[usize] {
        &self.n_bins_per_feature
    }

    /// Return the requested number of bins.
    #[must_use]
    pub fn n_bins(&self) -> usize {
        self.n_bins
    }

    /// Return the encoding method.
    #[must_use]
    pub fn encode(&self) -> BinEncoding {
        self.encode
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Assign a value to a bin index given sorted bin edges.
fn assign_bin<F: Float>(value: F, edges: &[F]) -> usize {
    let n_bins = edges.len() - 1;
    if n_bins == 0 {
        return 0;
    }
    // Binary search for the bin
    for (i, edge) in edges.iter().enumerate().skip(1) {
        if value < *edge {
            return i - 1;
        }
    }
    // Last bin for values >= last edge
    n_bins - 1
}

/// 1-D k-means bin edges, faithfully replicating scikit-learn 1.5.2's
/// `KBinsDiscretizer(strategy="kmeans")` path (`sklearn/preprocessing/_discretization.py:285-300`).
///
/// sklearn runs ONE `KMeans(n_clusters=n_bins, init=uniform-bin-centers, n_init=1)`
/// Lloyd fit on the column, sorts the resulting centers, and builds
/// `bin_edges = np.r_[col_min, (centers[1:]+centers[:-1])*0.5, col_max]`.
///
/// This reproduces the full Lloyd machinery used by `KMeans.fit`
/// (`sklearn/cluster/_kmeans.py` + `_k_means_lloyd.pyx` + `_k_means_common.pyx`):
///
/// - **Mean-centering** (`_kmeans.py:1486-1493,1543-1546`): `KMeans.fit` subtracts
///   `X.mean(axis=0)` from both the data and the init "for more accurate distance
///   computations", runs Lloyd on the centered data, then adds the mean back to the
///   centers. The distance argmin is computed via `||C||² - 2·x·C` which is NOT
///   translation-invariant in floating point, so this shift is load-bearing for
///   the converged local optimum (not just numerical hygiene).
/// - **Assignment** (`_k_means_lloyd.pyx:196-213`): each point goes to the center
///   minimizing `pairwise[j] = ||C_j||² - 2·x·C_j` (the `x²` term is dropped since it
///   is constant per point); ties resolve to the LOWEST center index (strict `<`).
/// - **Center update** (`_k_means_lloyd.pyx:215-218`, `_k_means_common.pyx:_average_centers`):
///   each new center is the mean of its assigned points.
/// - **Empty-cluster relocation** (`_k_means_common.pyx:_relocate_empty_clusters_dense`,
///   `:170-214`): for each empty cluster, take the points FARTHEST from their own
///   assigned center (largest squared distance, descending) and move one into the
///   empty cluster; the donor loses it. Skipped when `max(distances) == 0` (more
///   clusters than distinct samples). Any cluster still empty after relocation is
///   placed at the location of the heaviest cluster (`_average_centers` else-branch).
/// - **Convergence** (`_kmeans.py:704-755`): stop on strict convergence (no label
///   changed) OR when `center_shift_total = Σ_j (C_new[j] - C_old[j])² <= tol`, with
///   `tol = mean(var(column)) * 1e-4` (`_tolerance`, `_kmeans.py:286-294`,
///   population variance), OR `max_iter = 300`.
///
/// All intermediate arithmetic is done in `f64` (matching numpy's float64 default)
/// regardless of `F`, then converted back to `F` for the edges.
fn kmeans_1d<F: Float>(values: &[F], n_bins: usize) -> Vec<F> {
    let n = values.len();
    // Column min/max in F (the outer edges; sklearn uses the un-centered col_min/col_max).
    let min_v = values
        .iter()
        .copied()
        .fold(F::infinity(), num_traits::Float::min);
    let max_v = values
        .iter()
        .copied()
        .fold(F::neg_infinity(), num_traits::Float::max);

    if n == 0 || n_bins == 0 {
        // Degenerate: fall back to a uniform partition over [min, max].
        return (0..=n_bins)
            .map(|i| {
                min_v
                    + (max_v - min_v) * F::from(i).unwrap_or_else(F::zero)
                        / fdiv_or_one::<F>(n_bins)
            })
            .collect();
    }

    // Work entirely in f64 (numpy float64 default).
    let col: Vec<f64> = values.iter().map(|&v| v.to_f64().unwrap_or(0.0)).collect();
    let col_min = col.iter().copied().fold(f64::INFINITY, f64::min);
    let col_max = col.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    // Variance-scaled tolerance: tol = mean(var(column)) * 1e-4 (population variance,
    // ddof=0), == sklearn `_tolerance(X, 1e-4)` (`_kmeans.py:286-294`).
    let mean_all: f64 = col.iter().sum::<f64>() / (n as f64);
    let var: f64 = col
        .iter()
        .map(|&x| (x - mean_all) * (x - mean_all))
        .sum::<f64>()
        / (n as f64);
    let tol = var * 1e-4;

    // KMeans.fit mean-centers X and the init (`_kmeans.py:1486-1493`).
    let x_mean = mean_all;
    let xc: Vec<f64> = col.iter().map(|&x| x - x_mean).collect();

    // Uniform-bin-centers init (`_discretization.py:289-290`), then shifted by -x_mean.
    let mut centers: Vec<f64> = (0..n_bins)
        .map(|i| {
            let lo = col_min + (col_max - col_min) * (i as f64) / (n_bins as f64);
            let hi = col_min + (col_max - col_min) * ((i + 1) as f64) / (n_bins as f64);
            (lo + hi) * 0.5 - x_mean
        })
        .collect();

    let mut labels = vec![usize::MAX; n];
    let mut labels_old = vec![usize::MAX; n];
    let max_iter = 300usize;

    for _ in 0..max_iter {
        let centers_old = centers.clone();

        // --- Assignment: argmin_j (||C_j||² - 2·x·C_j), ties -> lowest index. ---
        let csq: Vec<f64> = centers_old.iter().map(|&c| c * c).collect();
        for i in 0..n {
            let xi = xc[i];
            let mut best_j = 0usize;
            let mut best = csq[0] - 2.0 * xi * centers_old[0];
            for j in 1..n_bins {
                let d = csq[j] - 2.0 * xi * centers_old[j];
                if d < best {
                    best = d;
                    best_j = j;
                }
            }
            labels[i] = best_j;
        }

        // --- Accumulate per-cluster sum and weight (count). ---
        let mut acc = vec![0.0f64; n_bins];
        let mut wic = vec![0.0f64; n_bins];
        for i in 0..n {
            acc[labels[i]] += xc[i];
            wic[labels[i]] += 1.0;
        }

        // --- Empty-cluster relocation (`_relocate_empty_clusters_dense`). ---
        let empty: Vec<usize> = (0..n_bins).filter(|&j| wic[j] == 0.0).collect();
        if !empty.is_empty() {
            // distances[i] = (xc[i] - centers_old[labels[i]])²
            let distances: Vec<f64> = (0..n)
                .map(|i| {
                    let d = xc[i] - centers_old[labels[i]];
                    d * d
                })
                .collect();
            let max_dist = distances.iter().copied().fold(0.0f64, f64::max);
            if max_dist != 0.0 {
                let n_empty = empty.len();
                // far_from_centers: the n_empty points with the largest distance,
                // in descending order. sklearn uses
                // `np.argpartition(distances, -n_empty)[:-n_empty-1:-1]`
                // (`_k_means_common.pyx:190`), whose introselect partition + reverse
                // slice resolves equal-distance ties toward the HIGHEST original index
                // (e.g. distances `[.04,.04,0,0,.04,.04]`, n_empty=2 -> far `[5, 4]`).
                // Match that by breaking distance ties on DESCENDING index, so the
                // duplicate-heavy `n_bins > n_distinct` relocation dumps the same
                // points onto the empty clusters as sklearn (the centers then coincide
                // and collapse under small-bin removal to sklearn's `n_bins_`).
                let mut order: Vec<usize> = (0..n).collect();
                order.sort_by(|&a, &b| {
                    distances[b]
                        .partial_cmp(&distances[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(b.cmp(&a))
                });
                for idx in 0..n_empty {
                    let new_cluster = empty[idx];
                    let far = order[idx];
                    let old_cluster = labels[far];
                    acc[old_cluster] -= xc[far];
                    acc[new_cluster] = xc[far];
                    wic[new_cluster] = 1.0;
                    wic[old_cluster] -= 1.0;
                }
            }
        }

        // --- Average; clusters still empty -> location of the heaviest cluster
        //     (`_average_centers` else-branch). ---
        let mut argmax_w = 0usize;
        for j in 1..n_bins {
            if wic[j] > wic[argmax_w] {
                argmax_w = j;
            }
        }
        for j in 0..n_bins {
            if wic[j] > 0.0 {
                centers[j] = acc[j] / wic[j];
            } else if wic[argmax_w] > 0.0 {
                centers[j] = acc[argmax_w] / wic[argmax_w];
            } else {
                centers[j] = centers_old[j];
            }
        }

        // --- Convergence (`_kmeans.py:724-739`). ---
        if labels == labels_old {
            // Strict convergence: no label changed.
            break;
        }
        let center_shift_tot: f64 = (0..n_bins)
            .map(|j| {
                let d = centers[j] - centers_old[j];
                d * d
            })
            .sum();
        if center_shift_tot <= tol {
            break;
        }
        labels_old.copy_from_slice(&labels);
    }

    // Shift centers back (`_kmeans.py:1546`) and sort (`_discretization.py:298`).
    let mut centers_out: Vec<f64> = centers.iter().map(|&c| c + x_mean).collect();
    centers_out.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // edges = [col_min, midpoints.., col_max] (`_discretization.py:299-300`).
    let mut edges = Vec::with_capacity(n_bins + 1);
    edges.push(min_v);
    for i in 0..n_bins.saturating_sub(1) {
        let mid = (centers_out[i] + centers_out[i + 1]) * 0.5;
        edges.push(F::from(mid).unwrap_or(min_v));
    }
    edges.push(max_v);

    edges
}

/// `F::from(n)` as a divisor, falling back to `F::one()` when `n == 0` to avoid a
/// division by zero in the degenerate fallback path.
fn fdiv_or_one<F: Float>(n: usize) -> F {
    if n == 0 {
        F::one()
    } else {
        F::from(n).unwrap_or_else(F::one)
    }
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for KBinsDiscretizer<F> {
    type Fitted = FittedKBinsDiscretizer<F>;
    type Error = FerroError;

    /// Fit by computing bin edges for each feature.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InsufficientSamples`] if the input has fewer than 2 rows.
    /// - [`FerroError::InvalidParameter`] if `n_bins` < 2.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedKBinsDiscretizer<F>, FerroError> {
        let n_samples = x.nrows();
        if n_samples < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: n_samples,
                context: "KBinsDiscretizer::fit".into(),
            });
        }
        if self.n_bins < 2 {
            return Err(FerroError::InvalidParameter {
                name: "n_bins".into(),
                reason: "n_bins must be at least 2".into(),
            });
        }

        let n_features = x.ncols();
        let mut bin_edges = Vec::with_capacity(n_features);
        let mut n_bins_per_feature = Vec::with_capacity(n_features);

        for j in 0..n_features {
            let mut col_vals: Vec<F> = x.column(j).iter().copied().collect();
            col_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let min_val = col_vals[0];
            let max_val = col_vals[col_vals.len() - 1];

            // Constant feature (sklearn :262-268): collapse to a single bin
            // spanning [-inf, +inf] so transform maps every value to bin 0.
            if min_val == max_val {
                bin_edges.push(vec![F::neg_infinity(), F::infinity()]);
                n_bins_per_feature.push(1);
                continue;
            }

            let edges: Vec<F> = match self.strategy {
                BinStrategy::Uniform => (0..=self.n_bins)
                    .map(|i| {
                        min_val
                            + (max_val - min_val) * F::from(i).unwrap()
                                / F::from(self.n_bins).unwrap()
                    })
                    .collect(),
                BinStrategy::Quantile => {
                    let n = col_vals.len();
                    (0..=self.n_bins)
                        .map(|i| {
                            let frac = F::from(i).unwrap() / F::from(self.n_bins).unwrap();
                            let pos = frac * F::from(n.saturating_sub(1)).unwrap();
                            let lo = pos.floor().to_usize().unwrap_or(0).min(n - 1);
                            let hi = pos.ceil().to_usize().unwrap_or(0).min(n - 1);
                            let f = pos - F::from(lo).unwrap();
                            col_vals[lo] * (F::one() - f) + col_vals[hi] * f
                        })
                        .collect()
                }
                BinStrategy::KMeans => kmeans_1d(&col_vals, self.n_bins),
            };

            // Small-bin removal for quantile and kmeans only (sklearn
            // :302-312): keep the first edge, then keep each subsequent edge
            // only if its gap to the previously kept edge exceeds 1e-8.
            // Uniform is never collapsed.
            match self.strategy {
                BinStrategy::Quantile | BinStrategy::KMeans => {
                    let tol = F::from(1e-8).unwrap_or_else(F::epsilon);
                    let mut kept: Vec<F> = Vec::with_capacity(edges.len());
                    for &edge in &edges {
                        match kept.last() {
                            None => kept.push(edge),
                            Some(&last) if edge - last > tol => kept.push(edge),
                            Some(_) => {}
                        }
                    }
                    n_bins_per_feature.push(kept.len() - 1);
                    bin_edges.push(kept);
                }
                BinStrategy::Uniform => {
                    n_bins_per_feature.push(self.n_bins);
                    bin_edges.push(edges);
                }
            }
        }

        Ok(FittedKBinsDiscretizer {
            bin_edges,
            n_bins_per_feature,
            n_bins: self.n_bins,
            encode: self.encode,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedKBinsDiscretizer<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Discretize features into bin indices or one-hot vectors.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns differs
    /// from the number of features seen during fitting.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features = self.bin_edges.len();
        if x.ncols() != n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedKBinsDiscretizer::transform".into(),
            });
        }

        let n_samples = x.nrows();

        match self.encode {
            BinEncoding::Ordinal => {
                let mut out = Array2::zeros((n_samples, n_features));
                for j in 0..n_features {
                    let edges = &self.bin_edges[j];
                    for i in 0..n_samples {
                        let bin = assign_bin(x[[i, j]], edges);
                        out[[i, j]] = F::from(bin).unwrap_or_else(F::zero);
                    }
                }
                Ok(out)
            }
            BinEncoding::OneHot => {
                // Output width is the sum of the per-feature bin counts, and
                // feature `j`'s columns start at the cumulative sum of the
                // preceding features' bin counts (sklearn one-hot over
                // `n_bins_`).
                let mut offsets = Vec::with_capacity(n_features + 1);
                let mut acc = 0usize;
                for &nb in &self.n_bins_per_feature {
                    offsets.push(acc);
                    acc += nb;
                }
                let n_out = acc;
                let mut out = Array2::zeros((n_samples, n_out));
                for j in 0..n_features {
                    let edges = &self.bin_edges[j];
                    let col_offset = offsets[j];
                    for i in 0..n_samples {
                        let bin = assign_bin(x[[i, j]], edges);
                        out[[i, col_offset + bin]] = F::one();
                    }
                }
                Ok(out)
            }
        }
    }
}

/// Implement `Transform` on the unfitted discretizer.
impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for KBinsDiscretizer<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Always returns an error — must be fitted first.
    fn transform(&self, _x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        Err(FerroError::InvalidParameter {
            name: "KBinsDiscretizer".into(),
            reason: "discretizer must be fitted before calling transform; use fit() first".into(),
        })
    }
}

impl<F: Float + Send + Sync + 'static> FitTransform<Array2<F>> for KBinsDiscretizer<F> {
    type FitError = FerroError;

    /// Fit and transform in one step.
    ///
    /// # Errors
    ///
    /// Returns an error if fitting fails.
    fn fit_transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let fitted = self.fit(x, &())?;
        fitted.transform(x)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_kbins_ordinal_uniform() {
        let disc = KBinsDiscretizer::<f64>::new(3, BinEncoding::Ordinal, BinStrategy::Uniform);
        let x = array![[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]];
        let fitted = disc.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        assert_eq!(out.ncols(), 1);
        // Check bin assignments
        assert_abs_diff_eq!(out[[0, 0]], 0.0, epsilon = 1e-10); // 0.0 → bin 0
        assert_abs_diff_eq!(out[[5, 0]], 2.0, epsilon = 1e-10); // 5.0 → bin 2 (last)
    }

    #[test]
    fn test_kbins_onehot_uniform() {
        let disc = KBinsDiscretizer::<f64>::new(3, BinEncoding::OneHot, BinStrategy::Uniform);
        let x = array![[0.0], [2.5], [5.0]];
        let fitted = disc.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        // 3 bins → 3 columns per feature
        assert_eq!(out.ncols(), 3);
        // Each row should have exactly one 1.0
        for i in 0..out.nrows() {
            let row_sum: f64 = out.row(i).iter().sum();
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_kbins_quantile_strategy() {
        let disc = KBinsDiscretizer::<f64>::new(4, BinEncoding::Ordinal, BinStrategy::Quantile);
        let x = array![[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]];
        let fitted = disc.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        // All values should be valid bin indices
        for v in &out {
            assert!(*v >= 0.0 && *v < 4.0);
        }
    }

    #[test]
    fn test_kbins_kmeans_strategy() {
        let disc = KBinsDiscretizer::<f64>::new(3, BinEncoding::Ordinal, BinStrategy::KMeans);
        let x = array![
            [0.0],
            [0.1],
            [0.2],
            [5.0],
            [5.1],
            [5.2],
            [10.0],
            [10.1],
            [10.2]
        ];
        let fitted = disc.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        // Values should be valid bin indices
        for v in &out {
            assert!(*v >= 0.0 && *v < 3.0);
        }
    }

    #[test]
    fn test_kbins_multi_feature() {
        let disc = KBinsDiscretizer::<f64>::new(3, BinEncoding::Ordinal, BinStrategy::Uniform);
        let x = array![[0.0, 10.0], [2.5, 15.0], [5.0, 20.0]];
        let fitted = disc.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        assert_eq!(out.ncols(), 2);
    }

    #[test]
    fn test_kbins_bin_edges() {
        let disc = KBinsDiscretizer::<f64>::new(3, BinEncoding::Ordinal, BinStrategy::Uniform);
        let x = array![[0.0], [3.0], [6.0]];
        let fitted = disc.fit(&x, &()).unwrap();
        let edges = &fitted.bin_edges()[0];
        // 4 edges for 3 bins: [0, 2, 4, 6]
        assert_eq!(edges.len(), 4);
        assert_abs_diff_eq!(edges[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(edges[3], 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_kbins_fit_transform() {
        let disc = KBinsDiscretizer::<f64>::new(3, BinEncoding::Ordinal, BinStrategy::Uniform);
        let x = array![[0.0], [2.5], [5.0]];
        let out = disc.fit_transform(&x).unwrap();
        assert_eq!(out.ncols(), 1);
    }

    #[test]
    fn test_kbins_insufficient_samples_error() {
        let disc = KBinsDiscretizer::<f64>::new(3, BinEncoding::Ordinal, BinStrategy::Uniform);
        let x = array![[1.0]];
        assert!(disc.fit(&x, &()).is_err());
    }

    #[test]
    fn test_kbins_too_few_bins_error() {
        let disc = KBinsDiscretizer::<f64>::new(1, BinEncoding::Ordinal, BinStrategy::Uniform);
        let x = array![[0.0], [1.0]];
        assert!(disc.fit(&x, &()).is_err());
    }

    #[test]
    fn test_kbins_shape_mismatch_error() {
        let disc = KBinsDiscretizer::<f64>::new(3, BinEncoding::Ordinal, BinStrategy::Uniform);
        let x_train = array![[0.0, 1.0], [2.0, 3.0]];
        let fitted = disc.fit(&x_train, &()).unwrap();
        let x_bad = array![[1.0, 2.0, 3.0]];
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_kbins_unfitted_error() {
        let disc = KBinsDiscretizer::<f64>::new(3, BinEncoding::Ordinal, BinStrategy::Uniform);
        let x = array![[0.0]];
        assert!(disc.transform(&x).is_err());
    }

    #[test]
    fn test_kbins_default() {
        let disc = KBinsDiscretizer::<f64>::default();
        assert_eq!(disc.n_bins(), 5);
        assert_eq!(disc.encode(), BinEncoding::Ordinal);
        assert_eq!(disc.strategy(), BinStrategy::Uniform);
    }

    #[test]
    fn test_kbins_ordinal_values_in_range() {
        let disc = KBinsDiscretizer::<f64>::new(5, BinEncoding::Ordinal, BinStrategy::Uniform);
        let x = array![[0.0], [2.5], [5.0], [7.5], [10.0]];
        let fitted = disc.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        for v in &out {
            assert!(*v >= 0.0 && *v < 5.0, "Bin index {v} out of range");
        }
    }
}
