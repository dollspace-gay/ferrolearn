//! Mean Shift clustering.
//!
//! This module provides [`MeanShift`], a non-parametric clustering algorithm
//! that finds the modes (local maxima) of the underlying kernel density
//! estimate.  Each data point is iteratively shifted toward the mean of all
//! points within a sphere of radius `bandwidth` until convergence.  After
//! convergence, nearby modes are merged into a single cluster center, and each
//! training point is assigned to the nearest center.
//!
//! # Algorithm
//!
//! 1. **Bandwidth estimation** (when `bandwidth` is `None`): use scikit-learn's
//!    kNN heuristic — for each point take the distance to its
//!    `k = int(n_samples * quantile)` nearest neighbour (`quantile = 0.3`,
//!    `k` clipped to `>= 1`) and average those per-point maxima over all
//!    points.
//! 2. **Mean shift iteration**: for each data point (candidate mode), compute
//!    the mean of all points within `bandwidth` distance and shift the
//!    candidate to that mean.  Repeat until the shift is smaller than `tol`
//!    or `max_iter` is reached.
//! 3. **Mode merging**: candidates whose final positions are within
//!    `bandwidth` of each other are merged (the one with more nearby points
//!    becomes the representative center).
//! 4. **Label assignment**: each training point is assigned to the nearest
//!    cluster center.
//!
//! Mean Shift does **not** require specifying the number of clusters ahead of
//! time; that number emerges from the data density and the bandwidth.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_cluster::MeanShift;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::Array2;
//!
//! let x = Array2::from_shape_vec((6, 2), vec![
//!     1.0_f64, 1.0,  1.1, 1.0,  1.0, 1.1,
//!     9.0, 9.0,  9.1, 9.0,  9.0, 9.1,
//! ]).unwrap();
//!
//! let model = MeanShift::<f64>::new().with_bandwidth(3.0);
//! let fitted = model.fit(&x, &()).unwrap();
//! assert_eq!(fitted.n_clusters(), 2);
//! ```
//!
//! # `## REQ status`
//!
//! Binary (R-DEFER-2), translating `sklearn/cluster/_mean_shift.py`
//! (`class MeanShift(ClusterMixin, BaseEstimator)` `:302-578`; `estimate_bandwidth`
//! `:43-106`). Design doc: `.design/cluster/mean_shift.md`. Cites use ferrolearn
//! symbol anchors / sklearn `file:line` (commit 156ef14); expected values from the
//! live sklearn 1.5.2 oracle (R-CHAR-3). Verify-and-document unit: the
//! explicit-bandwidth `labels_` PARTITION (up to a label permutation, well-separated
//! regime) and the `estimate_bandwidth` kNN VALUE (default no-subsampling path) match
//! sklearn and SHIP through the crate re-export. The `cluster_centers_` VALUES, the
//! `labels_` INTEGERS, and `n_iter_` DIVERGE — ferrolearn averages each merged mode
//! group in seed order, while sklearn retains the actual highest-intensity converged
//! mode sorted by intensity (#984/#986). There is no CPython binding (no
//! `RsMeanShift`; #993).
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (explicit-bandwidth `labels_` PARTITION up-to-permutation, separable data) | SHIPPED | impl `fn fit` (seed-per-point `mean_shift_single` + the mode-merge loop) recovers sklearn's grouping for an explicit `bandwidth` on well-separated data. Consumer: crate re-export `pub use mean_shift::{FittedMeanShift, MeanShift}` (`lib.rs`). Guards: `green_two_blobs_partition_matches_sklearn_bw2`, `green_three_blobs_partition_matches_sklearn_bw15`, `green_fresh_three_blobs_partition_matches_sklearn_bw3` in `tests/divergence_mean_shift.rs` (canonicalized, live-oracle). Underclaim: PARTITION up-to-permutation only — absolute `labels_` integers (REQ-4) + `cluster_centers_` values (REQ-2) diverge. |
//! | REQ-3 (`estimate_bandwidth` kNN heuristic, default path) | SHIPPED | impl `pub fn estimate_bandwidth(x, quantile)` mirrors sklearn `estimate_bandwidth(X, quantile=0.3)` (`_mean_shift.py:95-106`): `k=int(n*quantile)` (clip `>=1`), per-point max distance among the `k` nearest (self included at 0), averaged over points. Consumer: `fn fit` None-bandwidth path calls `estimate_bandwidth(x, 0.3)`. Guard: `divergence_auto_bandwidth_n_clusters` (`MeanShift::new().fit(fresh 3-blob)` → 3 clusters, matching sklearn; was 1 under the old median-pairwise). Fixed in #985. Caveat: the `n_samples`/`random_state` subsampling + `n_jobs` params are NOT implemented — they only affect the large-data approximation, not the default (full-sample) VALUE, which matches. |
//! | REQ-2 (`cluster_centers_` VALUE + intensity ordering) | NOT-STARTED | open prereq blocker #984. sklearn sorts converged modes by `(intensity, coord)` desc and retains the actual highest-intensity mode per `bandwidth`-ball (`_mean_shift.py:529-546`); `cluster_centers_` on docs = `[[3.333,6.0],[1.333,0.666]]`. ferrolearn `fn fit` averages each merged group in seed order — different values + ordering. |
//! | REQ-4 (`labels_` VALUE parity) | NOT-STARTED | open prereq blocker #986. sklearn `labels_` index into intensity-sorted `cluster_centers_` (`_mean_shift.py:548-557`), docs = `[1,1,1,0,0,0]`; ferrolearn assigns nearest of seed-ordered group-mean centers (same partition, permuted integers). Gated on REQ-2. |
//! | REQ-5 (ctor surface `seeds`/`bin_seeding`/`min_bin_freq`/`cluster_all`/`n_jobs`; drop non-sklearn `tol`) | NOT-STARTED | open prereq blocker #987. sklearn `__init__` (`_mean_shift.py:449-466`) = `bandwidth,seeds,bin_seeding,min_bin_freq,cluster_all,n_jobs,max_iter`; ferrolearn `MeanShift<F>` = `bandwidth/max_iter/tol` (missing 5; `tol` is non-sklearn — sklearn hardcodes `1e-3*bandwidth`). |
//! | REQ-6 (`cluster_all=False` orphan `-1`) | NOT-STARTED | open prereq blocker #988. sklearn fills `labels=-1` then assigns only within-`bandwidth` points (`_mean_shift.py:552-557`); ferrolearn `labels_: Array1<usize>` cannot hold `-1` and always assigns nearest. Needs a signed label type + `cluster_all`. |
//! | REQ-7 (`bin_seeding`/`get_bin_seeds`/`min_bin_freq` seeding) | NOT-STARTED | open prereq blocker #989. sklearn `get_bin_seeds(X, bandwidth, min_bin_freq)` bins to a `bandwidth`-grid (`_mean_shift.py:249-299`, `:491-495`); ferrolearn always seeds from every data point. |
//! | REQ-8 (convergence stop-threshold `1e-3*bandwidth`) | NOT-STARTED | open prereq blocker #990. sklearn `stop_thresh = 1e-3 * bandwidth` (`_mean_shift.py:113`, `:124-127`); ferrolearn stops at `shift < tol` with absolute default `tol=1e-3` (diverges for `bandwidth != 1`); `tol` is a non-sklearn knob. |
//! | REQ-9 (`n_iter_` semantics) | NOT-STARTED | open prereq blocker #991. sklearn `n_iter_ = max(completed_iterations)` incremented after the convergence/`max_iter` check (`_mean_shift.py:124-129`, `:514`), docs = `2`; ferrolearn `fn fit` takes `max(iter+1)` (loop counter, off-by-one/different convention). |
//! | REQ-10 (error ABI `InvalidParameterError` + no-neighbour `ValueError`) | NOT-STARTED | open prereq blocker #992. sklearn rejects `bandwidth<=0` with `InvalidParameterError` (`(0,inf)`, `_mean_shift.py:440`) and raises `ValueError` when no seed has neighbours (`:516-522`); ferrolearn raises `FerroError::InvalidParameter` (matching bound, different type/ABI) and cannot hit the no-neighbour case. |
//! | REQ-11 (PyO3 binding) | NOT-STARTED | open prereq blocker #993. `grep MeanShift ferrolearn-python/` empty — `RsKMeans` is registered but no `RsMeanShift`, so `import ferrolearn` cannot reach `MeanShift`. Only consumer is the crate re-export. |
//! | REQ-12 (ferray substrate) | NOT-STARTED | open prereq blocker #994. `mean_shift.rs` imports `ndarray::{Array1, Array2}` + `num_traits::Float`, not `ferray-core`/`ferray::linalg` (R-SUBSTRATE-1/2). |

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2};
use num_traits::Float;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration struct
// ─────────────────────────────────────────────────────────────────────────────

/// Mean Shift clustering configuration (unfitted).
///
/// Holds hyperparameters.  Call [`Fit::fit`] to run the algorithm and produce
/// a [`FittedMeanShift`].
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct MeanShift<F> {
    /// Kernel bandwidth (search radius).  When `None` the bandwidth is
    /// estimated via [`estimate_bandwidth`] (scikit-learn's kNN heuristic with
    /// `quantile = 0.3`).
    pub bandwidth: Option<F>,
    /// Maximum number of mean-shift iterations per seed point.
    pub max_iter: usize,
    /// Convergence tolerance: stop iterating when the shift magnitude is
    /// smaller than this value.
    pub tol: F,
}

impl<F: Float> MeanShift<F> {
    /// Create a new `MeanShift` with default hyperparameters.
    ///
    /// Bandwidth is estimated automatically (`None`), `max_iter = 300`,
    /// `tol = 1e-3`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            bandwidth: None,
            max_iter: 300,
            tol: F::from(1e-3).unwrap_or_else(F::epsilon),
        }
    }

    /// Set the bandwidth explicitly.
    ///
    /// Must be positive.  Setting this overrides automatic estimation.
    #[must_use]
    pub fn with_bandwidth(mut self, bandwidth: F) -> Self {
        self.bandwidth = Some(bandwidth);
        self
    }

    /// Set the maximum number of mean-shift iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance.
    #[must_use]
    pub fn with_tol(mut self, tol: F) -> Self {
        self.tol = tol;
        self
    }
}

impl<F: Float> Default for MeanShift<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fitted struct
// ─────────────────────────────────────────────────────────────────────────────

/// Fitted Mean Shift model.
///
/// Stores the discovered cluster centers and the labels assigned to the
/// training data.  Implements [`Predict`] to assign new points to the nearest
/// center.
#[derive(Debug, Clone)]
pub struct FittedMeanShift<F> {
    /// Cluster centers, shape `(n_clusters, n_features)`.
    cluster_centers_: Array2<F>,
    /// Cluster label for each training sample (0-indexed).
    labels_: Array1<usize>,
    /// Number of mean-shift iterations performed on the last seed point.
    n_iter_: usize,
}

impl<F: Float> FittedMeanShift<F> {
    /// Return the cluster centers, shape `(n_clusters, n_features)`.
    #[must_use]
    pub fn cluster_centers(&self) -> &Array2<F> {
        &self.cluster_centers_
    }

    /// Return the cluster labels for the training data.
    #[must_use]
    pub fn labels(&self) -> &Array1<usize> {
        &self.labels_
    }

    /// Return the number of mean-shift iterations in the last seed.
    #[must_use]
    pub fn n_iter(&self) -> usize {
        self.n_iter_
    }

    /// Return the number of clusters discovered.
    #[must_use]
    pub fn n_clusters(&self) -> usize {
        self.cluster_centers_.nrows()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Squared Euclidean distance between two equal-length slices.
#[inline]
fn sq_dist<F: Float>(a: &[F], b: &[F]) -> F {
    a.iter()
        .zip(b)
        .fold(F::zero(), |acc, (&ai, &bi)| acc + (ai - bi) * (ai - bi))
}

/// Estimate the mean-shift bandwidth with scikit-learn's kNN heuristic.
///
/// Mirrors `sklearn.cluster.estimate_bandwidth` (`_mean_shift.py:43-106`):
/// fit `NearestNeighbors(n_neighbors = int(n * quantile))` (clipped to `>= 1`)
/// on `x`, then return `sum over points of max(distance to its n_neighbors
/// nearest neighbours) / n_samples`. Because the query set is the SAME `x` the
/// neighbour index was fit on, each point's nearest-neighbour set includes
/// itself at distance `0`; the per-point maximum among the `k` nearest
/// (counting self) is therefore the element at index `k - 1` of that point's
/// ascending distance list. `quantile = 0.5` recovers (approximately) the
/// median of pairwise distances; the sklearn default is `quantile = 0.3`.
///
/// Returns an error if the dataset has fewer than 2 points.
pub fn estimate_bandwidth<F: Float>(x: &Array2<F>, quantile: F) -> Result<F, FerroError> {
    let n = x.nrows();
    if n < 2 {
        return Err(FerroError::InsufficientSamples {
            required: 2,
            actual: n,
            context: "MeanShift bandwidth estimation requires at least 2 samples".into(),
        });
    }

    // n_neighbors = int(n_samples * quantile), clipped to >= 1
    // (sklearn `_mean_shift.py:95-97`).
    let k = {
        let q = quantile.to_f64().unwrap_or(0.3);
        let raw = (n as f64 * q) as usize;
        raw.max(1)
    };

    // For each point, the max distance among its `k` nearest neighbours
    // (including itself at distance 0) is the element at index `k - 1` of the
    // ascending per-point distance list (sklearn `_mean_shift.py:101-106`).
    let mut sum = F::zero();
    for i in 0..n {
        let ri = x.row(i);
        let si = ri.as_slice().unwrap_or(&[]);
        let mut dists: Vec<F> = Vec::with_capacity(n);
        for j in 0..n {
            let rj = x.row(j);
            let sj = rj.as_slice().unwrap_or(&[]);
            dists.push(sq_dist(si, sj).sqrt());
        }
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sum = sum + dists[k - 1];
    }

    Ok(sum / F::from(n).unwrap_or_else(F::one))
}

/// Perform mean-shift iteration starting from `seed`.
///
/// Returns the converged mode position and the number of iterations taken.
fn mean_shift_single<F: Float>(
    seed: &[F],
    x: &Array2<F>,
    bw_sq: F,
    max_iter: usize,
    tol: F,
) -> (Vec<F>, usize) {
    let n_features = seed.len();
    let mut current = seed.to_vec();
    let mut n_iter = 0;

    for iter in 0..max_iter {
        n_iter = iter + 1;

        // Collect all points within bandwidth.
        let mut mean = vec![F::zero(); n_features];
        let mut count = F::zero();

        for i in 0..x.nrows() {
            let row = x.row(i);
            let rs = row.as_slice().unwrap_or(&[]);
            if sq_dist(&current, rs) <= bw_sq {
                for j in 0..n_features {
                    mean[j] = mean[j] + rs[j];
                }
                count = count + F::one();
            }
        }

        if count == F::zero() {
            // No neighbors — the point is isolated; keep it as-is.
            break;
        }

        for v in &mut mean {
            *v = *v / count;
        }

        // Compute shift magnitude.
        let shift = sq_dist(&current, &mean).sqrt();
        current = mean;

        if shift < tol {
            break;
        }
    }

    (current, n_iter)
}

// ─────────────────────────────────────────────────────────────────────────────
// Fit implementation
// ─────────────────────────────────────────────────────────────────────────────

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for MeanShift<F> {
    type Fitted = FittedMeanShift<F>;
    type Error = FerroError;

    /// Fit the Mean Shift model to the data.
    ///
    /// Each data point is used as a seed; all seeds are iteratively shifted
    /// toward the local mean until convergence.  Nearby converged modes are
    /// then merged into cluster centers.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InsufficientSamples`] if the dataset is empty.
    /// - [`FerroError::InvalidParameter`] if an explicit bandwidth ≤ 0 is
    ///   provided.
    /// - [`FerroError::NumericalInstability`] if bandwidth estimation or
    ///   center-array construction fails.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedMeanShift<F>, FerroError> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "MeanShift requires at least 1 sample".into(),
            });
        }

        // Resolve bandwidth.
        let bandwidth = match self.bandwidth {
            Some(bw) => {
                if bw <= F::zero() {
                    return Err(FerroError::InvalidParameter {
                        name: "bandwidth".into(),
                        reason: "must be positive".into(),
                    });
                }
                bw
            }
            None => estimate_bandwidth(x, F::from(0.3).unwrap_or_else(F::epsilon))?,
        };

        let bw_sq = bandwidth * bandwidth;

        // Run mean-shift from every data point as a seed.
        let mut modes: Vec<Vec<F>> = Vec::with_capacity(n_samples);
        let mut last_n_iter = 0usize;

        for i in 0..n_samples {
            let row = x.row(i);
            let seed = row.as_slice().unwrap_or(&[]);
            let (mode, n_iter) = mean_shift_single(seed, x, bw_sq, self.max_iter, self.tol);
            modes.push(mode);
            last_n_iter = last_n_iter.max(n_iter);
        }

        // Merge nearby modes into cluster centers.
        // We keep one representative per group of modes that lie within
        // `bandwidth` of each other.  We pick the first unclaimed mode as a
        // new center, then merge all other modes within `bandwidth` into it.
        let mut used = vec![false; modes.len()];
        let mut centers: Vec<Vec<F>> = Vec::new();

        for i in 0..modes.len() {
            if used[i] {
                continue;
            }
            used[i] = true;
            let mut group: Vec<&Vec<F>> = vec![&modes[i]];

            for j in (i + 1)..modes.len() {
                if !used[j] && sq_dist(&modes[i], &modes[j]).sqrt() < bandwidth {
                    used[j] = true;
                    group.push(&modes[j]);
                }
            }

            // Compute the mean of the group as the representative center.
            let mut center = vec![F::zero(); n_features];
            let g_len = F::from(group.len()).unwrap_or_else(F::one);
            for m in &group {
                for (k, &v) in m.iter().enumerate() {
                    center[k] = center[k] + v;
                }
            }
            for v in &mut center {
                *v = *v / g_len;
            }
            centers.push(center);
        }

        let n_centers = centers.len();

        // Build the center matrix.
        let flat: Vec<F> = centers.into_iter().flatten().collect();
        let cluster_centers =
            Array2::from_shape_vec((n_centers, n_features), flat).map_err(|_| {
                FerroError::NumericalInstability {
                    message: "failed to build cluster center matrix".into(),
                }
            })?;

        // Assign each training point to the nearest center.
        let mut labels = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let row = x.row(i);
            let rs = row.as_slice().unwrap_or(&[]);
            let mut best = 0usize;
            let mut best_dist = F::max_value();
            for c in 0..n_centers {
                let center_row = cluster_centers.row(c);
                let cs = center_row.as_slice().unwrap_or(&[]);
                let d = sq_dist(rs, cs);
                if d < best_dist {
                    best_dist = d;
                    best = c;
                }
            }
            labels[i] = best;
        }

        Ok(FittedMeanShift {
            cluster_centers_: cluster_centers,
            labels_: labels,
            n_iter_: last_n_iter,
        })
    }
}

impl<F: Float + Send + Sync + 'static> MeanShift<F> {
    /// Fit on `x` and return labels for those samples in one call.
    /// Equivalent to sklearn `ClusterMixin.fit_predict`.
    ///
    /// # Errors
    ///
    /// Forwards any error from [`Fit::fit`] / [`Predict::predict`].
    pub fn fit_predict(&self, x: &Array2<F>) -> Result<Array1<usize>, FerroError> {
        let fitted = self.fit(x, &())?;
        fitted.predict(x)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Predict implementation
// ─────────────────────────────────────────────────────────────────────────────

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedMeanShift<F> {
    type Output = Array1<usize>;
    type Error = FerroError;

    /// Assign each sample to the nearest cluster center.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does
    /// not match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<usize>, FerroError> {
        let n_features = x.ncols();
        let expected = self.cluster_centers_.ncols();
        if n_features != expected {
            return Err(FerroError::ShapeMismatch {
                expected: vec![expected],
                actual: vec![n_features],
                context: "number of features must match the fitted MeanShift model".into(),
            });
        }

        let n_samples = x.nrows();
        let n_centers = self.cluster_centers_.nrows();
        let mut labels = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let row = x.row(i);
            let rs = row.as_slice().unwrap_or(&[]);
            let mut best = 0usize;
            let mut best_dist = F::max_value();
            for c in 0..n_centers {
                let cr = self.cluster_centers_.row(c);
                let cs = cr.as_slice().unwrap_or(&[]);
                let d = sq_dist(rs, cs);
                if d < best_dist {
                    best_dist = d;
                    best = c;
                }
            }
            labels[i] = best;
        }

        Ok(labels)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Two well-separated 2-D blobs.
    fn two_blobs() -> Array2<f64> {
        Array2::from_shape_vec(
            (10, 2),
            vec![
                0.0, 0.0, 0.2, 0.1, -0.1, 0.2, 0.1, -0.1, 0.0, 0.1, 10.0, 10.0, 10.2, 10.1, 9.9,
                10.2, 10.1, 9.9, 10.0, 10.1,
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_two_blobs_correct_clusters() {
        let x = two_blobs();
        let model = MeanShift::<f64>::new().with_bandwidth(2.0);
        let fitted = model.fit(&x, &()).unwrap();

        assert_eq!(fitted.n_clusters(), 2);

        let labels = fitted.labels();
        // Points 0-4 should be in the same cluster.
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[0], labels[3]);
        assert_eq!(labels[0], labels[4]);
        // Points 5-9 should be in the same cluster.
        assert_eq!(labels[5], labels[6]);
        assert_eq!(labels[5], labels[7]);
        assert_eq!(labels[5], labels[8]);
        assert_eq!(labels[5], labels[9]);
        // Different clusters.
        assert_ne!(labels[0], labels[5]);
    }

    #[test]
    fn test_labels_length() {
        let x = two_blobs();
        let fitted = MeanShift::<f64>::new()
            .with_bandwidth(2.0)
            .fit(&x, &())
            .unwrap();
        assert_eq!(fitted.labels().len(), x.nrows());
    }

    #[test]
    fn test_cluster_centers_shape() {
        let x = two_blobs();
        let fitted = MeanShift::<f64>::new()
            .with_bandwidth(2.0)
            .fit(&x, &())
            .unwrap();
        let centers = fitted.cluster_centers();
        assert_eq!(centers.ncols(), 2);
        assert_eq!(centers.nrows(), 2);
    }

    #[test]
    fn test_centers_near_blob_means() {
        let x = two_blobs();
        let fitted = MeanShift::<f64>::new()
            .with_bandwidth(2.0)
            .fit(&x, &())
            .unwrap();

        let centers = fitted.cluster_centers();
        // The two centers should be approximately at (0.04, 0.06) and (10.04, 10.06).
        // Collect distances to (0,0) and (10,10).
        let near_origin =
            (0..centers.nrows()).any(|i| centers[[i, 0]].hypot(centers[[i, 1]]) < 1.0);
        let near_far = (0..centers.nrows())
            .any(|i| (centers[[i, 0]] - 10.0).hypot(centers[[i, 1]] - 10.0) < 1.0);

        assert!(near_origin, "expected a center near the origin cluster");
        assert!(near_far, "expected a center near the (10,10) cluster");
    }

    #[test]
    fn test_single_cluster_large_bandwidth() {
        let x = two_blobs();
        // Bandwidth large enough to merge everything.
        let fitted = MeanShift::<f64>::new()
            .with_bandwidth(50.0)
            .fit(&x, &())
            .unwrap();
        assert_eq!(fitted.n_clusters(), 1);
        for &l in fitted.labels() {
            assert_eq!(l, 0);
        }
    }

    #[test]
    fn test_auto_bandwidth_runs() {
        let x = two_blobs();
        // Auto bandwidth uses sklearn's kNN heuristic (estimate_bandwidth, quantile 0.3).
        let fitted = MeanShift::<f64>::new().fit(&x, &()).unwrap();
        // On these tight blobs the kNN bandwidth is small, so the run produces at
        // least one cluster; exact-count parity vs sklearn rides REQ-2/REQ-4 (#984/#986).
        assert!(fitted.n_clusters() >= 1);
    }

    #[test]
    fn test_predict_on_new_points() {
        let x = two_blobs();
        let fitted = MeanShift::<f64>::new()
            .with_bandwidth(2.0)
            .fit(&x, &())
            .unwrap();

        let new_x = Array2::from_shape_vec((2, 2), vec![0.05, 0.05, 10.05, 10.05]).unwrap();
        let preds = fitted.predict(&new_x).unwrap();

        // Each new point should get the same label as the nearby training points.
        let label_near_origin = fitted.labels()[0];
        let label_near_far = fitted.labels()[5];
        assert_eq!(preds[0], label_near_origin);
        assert_eq!(preds[1], label_near_far);
    }

    #[test]
    fn test_predict_shape_mismatch() {
        let x = two_blobs();
        let fitted = MeanShift::<f64>::new()
            .with_bandwidth(2.0)
            .fit(&x, &())
            .unwrap();

        let bad = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert!(fitted.predict(&bad).is_err());
    }

    #[test]
    fn test_single_point() {
        let x = Array2::from_shape_vec((1, 2), vec![3.0, 4.0]).unwrap();
        let fitted = MeanShift::<f64>::new()
            .with_bandwidth(1.0)
            .fit(&x, &())
            .unwrap();
        assert_eq!(fitted.n_clusters(), 1);
        assert_eq!(fitted.labels()[0], 0);
    }

    #[test]
    fn test_empty_data_error() {
        let x = Array2::<f64>::zeros((0, 2));
        let result = MeanShift::<f64>::new().with_bandwidth(1.0).fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_bandwidth_error() {
        let x = two_blobs();
        let result = MeanShift::<f64>::new().with_bandwidth(-1.0).fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_bandwidth_error() {
        let x = two_blobs();
        let result = MeanShift::<f64>::new().with_bandwidth(0.0).fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_n_iter_non_zero() {
        let x = two_blobs();
        let fitted = MeanShift::<f64>::new()
            .with_bandwidth(2.0)
            .fit(&x, &())
            .unwrap();
        assert!(fitted.n_iter() > 0);
    }

    #[test]
    fn test_f32_support() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                0.0f32, 0.0, 0.1, 0.1, -0.1, 0.1, 8.0, 8.0, 8.1, 8.1, 7.9, 8.1,
            ],
        )
        .unwrap();

        let fitted = MeanShift::<f32>::new()
            .with_bandwidth(2.0)
            .fit(&x, &())
            .unwrap();
        assert_eq!(fitted.labels().len(), 6);
        assert_eq!(fitted.n_clusters(), 2);
    }

    #[test]
    fn test_three_clusters() {
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 10.0, 0.0, 10.1, 0.0, 10.0, 0.1, 0.0, 10.0, 0.1,
                10.0, 0.0, 10.1,
            ],
        )
        .unwrap();

        let fitted = MeanShift::<f64>::new()
            .with_bandwidth(1.5)
            .fit(&x, &())
            .unwrap();
        assert_eq!(fitted.n_clusters(), 3);

        let labels = fitted.labels();
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[3], labels[5]);
        assert_eq!(labels[6], labels[7]);
        assert_eq!(labels[6], labels[8]);
        assert_ne!(labels[0], labels[3]);
        assert_ne!(labels[0], labels[6]);
        assert_ne!(labels[3], labels[6]);
    }

    #[test]
    fn test_identical_points_single_cluster() {
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        )
        .unwrap();
        let fitted = MeanShift::<f64>::new()
            .with_bandwidth(1.0)
            .fit(&x, &())
            .unwrap();
        assert_eq!(fitted.n_clusters(), 1);
    }

    #[test]
    fn test_center_coordinates_reasonable() {
        // Single tight cluster near (5, 5).
        let x =
            Array2::from_shape_vec((4, 2), vec![5.0, 5.0, 5.1, 4.9, 4.9, 5.1, 5.0, 5.0]).unwrap();
        let fitted = MeanShift::<f64>::new()
            .with_bandwidth(1.0)
            .fit(&x, &())
            .unwrap();
        assert_eq!(fitted.n_clusters(), 1);
        let c = fitted.cluster_centers();
        assert_relative_eq!(c[[0, 0]], 5.0, epsilon = 0.2);
        assert_relative_eq!(c[[0, 1]], 5.0, epsilon = 0.2);
    }

    #[test]
    fn test_predict_labels_range() {
        let x = two_blobs();
        let fitted = MeanShift::<f64>::new()
            .with_bandwidth(2.0)
            .fit(&x, &())
            .unwrap();
        let n_c = fitted.n_clusters();
        for &l in fitted.labels() {
            assert!(l < n_c);
        }
    }

    #[test]
    fn test_default_trait() {
        let model: MeanShift<f64> = MeanShift::default();
        assert!(model.bandwidth.is_none());
        assert_eq!(model.max_iter, 300);
    }
}
