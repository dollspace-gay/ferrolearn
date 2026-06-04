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
//! | REQ-2 | KMeans bin edges/transform on well-separated data | SHIPPED (scoped) | `kmeans_1d` deterministic uniform-midpoint init (`:289-290`) matches sklearn Lloyd on well-separated data (edges [0,2.6,7.6,10.2]); EXACT parity on degenerate/duplicate-heavy data (sklearn empty-cluster relocation) is NOT-STARTED — carve-out #1378 |
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
//! | REQ-14 | KMeans EXACT parity on degenerate/duplicate-heavy data (empty-cluster relocation) | NOT-STARTED | `kmeans_1d` keeps empty clusters; sklearn relocates (`cluster/_kmeans.py`) — carve-out blocker #1378 |

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

/// Simple 1D k-means to find bin edges.
fn kmeans_1d<F: Float>(values: &[F], n_bins: usize, max_iter: usize) -> Vec<F> {
    let n = values.len();
    if n <= n_bins || n_bins == 0 {
        // Fallback to uniform
        let min_v = values
            .iter()
            .copied()
            .fold(F::infinity(), num_traits::Float::min);
        let max_v = values
            .iter()
            .copied()
            .fold(F::neg_infinity(), num_traits::Float::max);
        return (0..=n_bins)
            .map(|i| min_v + (max_v - min_v) * F::from(i).unwrap() / F::from(n_bins).unwrap())
            .collect();
    }

    // Initialize centroids using uniform spacing
    let min_v = values
        .iter()
        .copied()
        .fold(F::infinity(), num_traits::Float::min);
    let max_v = values
        .iter()
        .copied()
        .fold(F::neg_infinity(), num_traits::Float::max);

    let mut centroids: Vec<F> = (0..n_bins)
        .map(|i| {
            min_v
                + (max_v - min_v) * (F::from(i).unwrap() + F::from(0.5).unwrap())
                    / F::from(n_bins).unwrap()
        })
        .collect();

    for _ in 0..max_iter {
        // Assign each value to nearest centroid
        let mut sums = vec![F::zero(); n_bins];
        let mut counts = vec![0usize; n_bins];

        for &v in values {
            let mut best_c = 0;
            let mut best_dist = F::infinity();
            for (c, &centroid) in centroids.iter().enumerate() {
                let d = (v - centroid).abs();
                if d < best_dist {
                    best_dist = d;
                    best_c = c;
                }
            }
            sums[best_c] = sums[best_c] + v;
            counts[best_c] += 1;
        }

        // Update centroids
        let mut converged = true;
        for c in 0..n_bins {
            if counts[c] > 0 {
                let new_centroid = sums[c] / F::from(counts[c]).unwrap();
                if (new_centroid - centroids[c]).abs() > F::from(1e-10).unwrap_or_else(F::epsilon) {
                    converged = false;
                }
                centroids[c] = new_centroid;
            }
        }
        if converged {
            break;
        }
    }

    // Sort centroids and compute edges as midpoints
    centroids.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut edges = Vec::with_capacity(n_bins + 1);
    edges.push(min_v);
    for i in 0..n_bins - 1 {
        let mid = (centroids[i] + centroids[i + 1]) / (F::one() + F::one());
        edges.push(mid);
    }
    edges.push(max_v);

    edges
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
                BinStrategy::KMeans => kmeans_1d(&col_vals, self.n_bins, 100),
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
