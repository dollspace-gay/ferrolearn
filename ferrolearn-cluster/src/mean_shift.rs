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
//!    candidate to that mean.  Repeat until the shift is `<= 1e-3 * bandwidth`
//!    (sklearn's `stop_thresh`) or `max_iter` is reached.
//! 3. **De-duplication** (sklearn post-processing): converged modes are sorted
//!    by `(intensity, coordinate)` descending — `intensity` being the number of
//!    original points within `bandwidth` of the final mode — then greedily
//!    thinned so that at most one mode survives per `bandwidth`-ball (the
//!    highest-intensity one). The retained `cluster_centers_` are the ACTUAL
//!    converged modes, in intensity-sorted order.
//! 4. **Label assignment**: each training point is assigned to the nearest
//!    cluster center (which indexes into the intensity-sorted order).
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
//! live sklearn 1.5.2 oracle (R-CHAR-3). The `cluster_centers_` VALUES (~1e-6), the
//! `labels_` INTEGERS (exact), and `n_iter_` now MATCH sklearn — `fn fit` sorts
//! converged modes by `(intensity, coord)` descending and retains the actual
//! highest-intensity mode per `bandwidth`-ball, and `mean_shift_single` uses
//! `stop_thresh = 1e-3 * bandwidth` (#984/#986/#990, n_iter side-effect closes #991).
//! There is no CPython binding (no `RsMeanShift`; #993).
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (explicit-bandwidth `labels_` PARTITION up-to-permutation, separable data) | SHIPPED | impl `fn fit` (seed-per-point `mean_shift_single` + the de-dup loop) recovers sklearn's grouping for an explicit `bandwidth` on well-separated data. Consumer: crate re-export `pub use mean_shift::{FittedMeanShift, MeanShift}` (`lib.rs`). Guards: `green_two_blobs_partition_matches_sklearn_bw2`, `green_three_blobs_partition_matches_sklearn_bw15`, `green_fresh_three_blobs_partition_matches_sklearn_bw3` in `tests/divergence_mean_shift.rs` (canonicalized, live-oracle). Now SUPERSEDED by the exact-value guards (REQ-2/REQ-4) below. |
//! | REQ-3 (`estimate_bandwidth` kNN heuristic, default path) | SHIPPED | impl `pub fn estimate_bandwidth(x, quantile)` mirrors sklearn `estimate_bandwidth(X, quantile=0.3)` (`_mean_shift.py:95-106`): `k=int(n*quantile)` (clip `>=1`), per-point max distance among the `k` nearest (self included at 0), averaged over points. Consumer: `fn fit` None-bandwidth path calls `estimate_bandwidth(x, 0.3)`. Guard: `divergence_auto_bandwidth_n_clusters` (`MeanShift::new().fit(fresh 3-blob)` → 3 clusters, matching sklearn; was 1 under the old median-pairwise). Fixed in #985. Caveat: the `n_samples`/`random_state` subsampling + `n_jobs` params are NOT implemented — they only affect the large-data approximation, not the default (full-sample) VALUE, which matches. |
//! | REQ-2 (`cluster_centers_` VALUE + intensity ordering) | SHIPPED | impl `fn fit` collects `(converged_mode, intensity)` from `mean_shift_single` over every seed, de-dups exact-equal modes (`modes_equal`), sorts by `(intensity, coord)` DESC (`coord_cmp` + intensity), then greedily retains the highest-intensity mode per `bandwidth`-ball (`_mean_shift.py:529-546`), keeping the ACTUAL converged mode in intensity-sorted order. Consumer: crate re-export `pub use mean_shift::{FittedMeanShift, MeanShift}` (`lib.rs`). Guards (live-oracle, ~1e-6): `green_docstring_centers_and_labels_match_sklearn_bw2` (docs → `[[3.333…,6.0],[1.333…,0.666…]]`), `green_two_blobs_centers_and_labels_match_sklearn_bw2`, `green_three_blobs_centers_and_labels_match_sklearn_bw15`, `green_two_clusters_centers_and_labels_match_sklearn_bw5`. |
//! | REQ-4 (`labels_` VALUE parity) | SHIPPED | impl `fn fit` label loop assigns each point to the nearest retained center (`cluster_all=True`, n_neighbors=1, `_mean_shift.py:548-553`); since the centers are now in sklearn's intensity-sorted order (REQ-2) the integers match exactly. Consumer: crate re-export (`lib.rs`). Guards (live-oracle, exact integers): same four `green_*_centers_and_labels_*` tests (docs → `[1,1,1,0,0,0]`; three_blobs bw1.5 → `[2,2,2,0,0,0,1,1,1]`). The signed `-1`/`cluster_all=False` path stays REQ-6 (#988). |
//! | REQ-5 (ctor surface `seeds`/`bin_seeding`/`min_bin_freq`/`cluster_all`/`n_jobs`; drop non-sklearn `tol`) | NOT-STARTED | open prereq blocker #987. sklearn `__init__` (`_mean_shift.py:449-466`) = `bandwidth,seeds,bin_seeding,min_bin_freq,cluster_all,n_jobs,max_iter`; ferrolearn `MeanShift<F>` = `bandwidth/max_iter/tol` (missing 5; `tol` is non-sklearn — sklearn hardcodes `1e-3*bandwidth`). |
//! | REQ-6 (`cluster_all=False` orphan `-1`) | NOT-STARTED | open prereq blocker #988. sklearn fills `labels=-1` then assigns only within-`bandwidth` points (`_mean_shift.py:552-557`); ferrolearn `labels_: Array1<usize>` cannot hold `-1` and always assigns nearest. Needs a signed label type + `cluster_all`. |
//! | REQ-7 (`bin_seeding`/`get_bin_seeds`/`min_bin_freq` seeding) | NOT-STARTED | open prereq blocker #989. sklearn `get_bin_seeds(X, bandwidth, min_bin_freq)` bins to a `bandwidth`-grid (`_mean_shift.py:249-299`, `:491-495`); ferrolearn always seeds from every data point. |
//! | REQ-8 (convergence stop-threshold `1e-3*bandwidth`) | SHIPPED | impl `fn mean_shift_single` sets `stop_thresh = 1e-3 * bandwidth` and breaks on `shift <= stop_thresh` (`_mean_shift.py:113`, `:124-127`), so the converged modes match sklearn's float VALUES for any bandwidth (this is what makes REQ-2 bit-exact). The non-sklearn `tol` field is now unused by `fit` (its removal is the ctor-surface REQ-5 / #987). Consumer: `fn fit` seed loop. Guards: the `bw != 1` value tests `green_three_blobs_centers_and_labels_match_sklearn_bw15` (bw=1.5) and `green_two_clusters_centers_and_labels_match_sklearn_bw5` (bw=5.0) — their centers match to ~1e-6 only with the scaled threshold. |
//! | REQ-9 (`n_iter_` semantics) | SHIPPED | side-effect of the REQ-8 fix: `fn mean_shift_single` now mirrors `_mean_shift_single_seed` exactly — increments `completed_iterations` AFTER the convergence/`max_iter` check and returns it (`_mean_shift.py:124-129`); `fn fit` takes `n_iter_max = max(completed)` (`:514`). Consumer: `fn fit`. Guard: `green_n_iter_matches_sklearn_docs_bw2` (docs bw=2 → sklearn `n_iter_ == 2`). |
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
    /// Maximum number of mean-shift iterations performed across all seeds
    /// (sklearn `n_iter_ = max([completed_iterations])`).
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

    /// Return the maximum number of mean-shift iterations across all seeds
    /// (sklearn `n_iter_`).
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

/// Exact equality of two converged-mode coordinate vectors.
///
/// Mirrors Python keying `center_intensity_dict[tuple(my_mean)]`
/// (`_mean_shift.py:512`): the dict key is the exact float tuple, so two seeds
/// collapse to one entry only when their converged modes are bit-equal. We use
/// exact `==` (not a tolerance) to match the `dict` semantics precisely.
#[inline]
fn modes_equal<F: Float>(a: &[F], b: &[F]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(&ai, &bi)| ai == bi)
}

/// Lexicographic comparison of two coordinate vectors, ASCENDING.
///
/// Mirrors Python's tuple ordering used in
/// `sorted(..., key=lambda t: (t[1], t[0]))` (`_mean_shift.py:531`): the
/// coordinate tuple `t[0]` is compared element-by-element, the first differing
/// element deciding the order. The caller applies the `reverse=True` direction.
/// NaN coordinates are impossible for a converged finite-data mean, so a
/// `partial_cmp` fallback to `Equal` is never hit on the real path.
#[inline]
fn coord_cmp<F: Float>(a: &[F], b: &[F]) -> std::cmp::Ordering {
    for (&ai, &bi) in a.iter().zip(b) {
        match ai.partial_cmp(&bi) {
            Some(std::cmp::Ordering::Equal) | None => {}
            Some(ord) => return ord,
        }
    }
    std::cmp::Ordering::Equal
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
/// Mirrors `_mean_shift_single_seed` (`sklearn/cluster/_mean_shift.py:110-130`):
/// at each step the candidate is moved to the (flat-kernel) mean of all points
/// within `bandwidth`, and the loop stops when either the L2 shift drops to
/// `stop_thresh = 1e-3 * bandwidth` (`:113`, `:124-127`) or `completed_iterations
/// == max_iter` (`:126`). sklearn checks convergence/`max_iter` immediately
/// *after* the mean update (`:124-128`) and increments `completed_iterations`
/// only when it has NOT converged (`:129`), so the returned count is the number
/// of *additional* steps taken after the very first mean.
///
/// Returns `(converged_mode, n_points_within_at_final_mode, completed_iterations)`.
/// `n_points_within` is the membership count at the FINAL mode (sklearn returns
/// `len(points_within)` from the last loop body that ran, `:130`).
fn mean_shift_single<F: Float>(
    seed: &[F],
    x: &Array2<F>,
    bandwidth: F,
    bw_sq: F,
    max_iter: usize,
) -> (Vec<F>, usize, usize) {
    let n_features = seed.len();
    let mut current = seed.to_vec();
    // sklearn: `stop_thresh = 1e-3 * bandwidth` (`_mean_shift.py:113`). NOT an
    // absolute `tol` — scales with the bandwidth so the converged modes match
    // sklearn's float values for any bandwidth (REQ-8).
    let stop_thresh = F::from(1e-3).unwrap_or_else(F::epsilon) * bandwidth;
    let mut completed_iterations = 0usize;
    let mut count_within = 0usize;

    loop {
        // Find mean of points within bandwidth of the current mean
        // (sklearn `_mean_shift.py:117-122`).
        let mut mean = vec![F::zero(); n_features];
        let mut count = F::zero();
        let mut n_within = 0usize;

        for i in 0..x.nrows() {
            let row = x.row(i);
            let rs = row.as_slice().unwrap_or(&[]);
            if sq_dist(&current, rs) <= bw_sq {
                for j in 0..n_features {
                    mean[j] = mean[j] + rs[j];
                }
                count = count + F::one();
                n_within += 1;
            }
        }

        if n_within == 0 {
            // No neighbours — sklearn `break`s without updating the mean or the
            // membership count (`_mean_shift.py:119-120`).
            break;
        }
        count_within = n_within;

        for v in &mut mean {
            *v = *v / count;
        }

        // L2 shift magnitude `||my_mean - my_old_mean||` (sklearn `:125`).
        let shift = sq_dist(&current, &mean).sqrt();
        current = mean;

        // Converged or at max_iter → break BEFORE incrementing
        // (sklearn `_mean_shift.py:124-129`).
        if shift <= stop_thresh || completed_iterations == max_iter {
            break;
        }
        completed_iterations += 1;
    }

    (current, count_within, completed_iterations)
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

        // Run mean-shift from EVERY data point as a seed (the sklearn default
        // `seeds=None` / `bin_seeding=False` path: `seeds = X`,
        // `_mean_shift.py:491-495`). Each seed climbs to a converged mode;
        // collect `(mode, intensity)` where `intensity` is the number of
        // original points within `bandwidth` of the FINAL mode
        // (`_mean_shift.py:130`).
        //
        // `center_intensity_dict[tuple(mean)] = n_points_within` only when
        // `n_points_within > 0` (`_mean_shift.py:510-512`). Because the dict is
        // keyed by the exact converged-mode value, duplicate modes collapse to
        // a single entry; the intensity is deterministic per mode so the kept
        // value matches. We replicate the dict semantics with an order-stable
        // de-dup keyed by the exact mode coordinates.
        //
        // `n_iter_ = max([completed_iterations])` over all seeds
        // (`_mean_shift.py:514`).
        let mut mode_keys: Vec<Vec<F>> = Vec::new();
        let mut intensities: Vec<usize> = Vec::new();
        let mut n_iter_max = 0usize;

        for i in 0..n_samples {
            let row = x.row(i);
            let seed = row.as_slice().unwrap_or(&[]);
            let (mode, intensity, completed) =
                mean_shift_single(seed, x, bandwidth, bw_sq, self.max_iter);
            n_iter_max = n_iter_max.max(completed);
            if intensity == 0 {
                // sklearn skips seeds whose final mode has no members
                // (`_mean_shift.py:511`); these never enter the dict.
                continue;
            }
            // Dict semantics: keep one entry per exact-equal mode value.
            if let Some(pos) = mode_keys.iter().position(|m| modes_equal(m, &mode)) {
                // Python's dict OVERWRITES with the same (deterministic)
                // intensity — a no-op for the value; the position is retained.
                intensities[pos] = intensity;
            } else {
                mode_keys.push(mode);
                intensities.push(intensity);
            }
        }

        if mode_keys.is_empty() {
            // Mirrors sklearn's "No point was within bandwidth of any seed"
            // `ValueError` (`_mean_shift.py:516-522`). Unreachable on the
            // seeds=X path (each seed is a data point, self-inclusive), but
            // kept for contract fidelity.
            return Err(FerroError::NumericalInstability {
                message: "no point was within bandwidth of any seed".into(),
            });
        }

        // POST-PROCESSING — remove near-duplicate modes
        // (`_mean_shift.py:524-546`).
        //
        // Sort by `(intensity, coordinate-tuple)` DESCENDING — sklearn:
        // `sorted(center_intensity_dict.items(), key=lambda t: (t[1], t[0]),
        // reverse=True)` (`:529-533`). Python's tuple comparison is
        // lexicographic and total; `reverse=True` orders intensity desc then,
        // on ties, coordinate-tuple desc. Python's `sorted` is stable, but the
        // key is a total order over distinct (mode) keys so stability is not
        // observable here.
        let mut order: Vec<usize> = (0..mode_keys.len()).collect();
        order.sort_by(|&a, &b| {
            // Descending: compare b vs a.
            intensities[b]
                .cmp(&intensities[a])
                .then_with(|| coord_cmp(&mode_keys[b], &mode_keys[a]))
        });
        let sorted_centers: Vec<&Vec<F>> = order.iter().map(|&i| &mode_keys[i]).collect();

        // Greedy radius-`bandwidth` unique selection over the sorted modes
        // (`_mean_shift.py:535-546`): walk the intensity-sorted modes; for each
        // still-unique mode, mark every mode within `bandwidth` (inclusive,
        // `radius_neighbors`) as non-unique, then re-mark the current one
        // unique. The kept modes are emitted IN THE SORTED-BY-INTENSITY ORDER
        // — that order is what `labels_` indexes into.
        let m = sorted_centers.len();
        let mut unique = vec![true; m];
        for i in 0..m {
            if unique[i] {
                for (j, item) in unique.iter_mut().enumerate() {
                    // `radius_neighbors([center])` returns points within
                    // `bandwidth` inclusive (`<= bandwidth`).
                    if sq_dist(sorted_centers[i], sorted_centers[j]) <= bw_sq {
                        *item = false;
                    }
                }
                unique[i] = true; // leave the current point as unique.
            }
        }

        let kept: Vec<&Vec<F>> = (0..m)
            .filter(|&i| unique[i])
            .map(|i| sorted_centers[i])
            .collect();
        let n_centers = kept.len();

        // Build the center matrix in the sorted-by-intensity order.
        let flat: Vec<F> = kept.iter().flat_map(|c| c.iter().copied()).collect();
        let cluster_centers =
            Array2::from_shape_vec((n_centers, n_features), flat).map_err(|_| {
                FerroError::NumericalInstability {
                    message: "failed to build cluster center matrix".into(),
                }
            })?;

        // ASSIGN LABELS — nearest cluster center, `n_neighbors=1`
        // (`_mean_shift.py:548-553`, `cluster_all=True`).
        let mut labels = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let row = x.row(i);
            let rs = row.as_slice().unwrap_or(&[]);
            let mut best = 0usize;
            let mut best_dist = F::max_value();
            for (c, center) in kept.iter().enumerate() {
                let d = sq_dist(rs, center);
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
            n_iter_: n_iter_max,
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
