//! DBSCAN (Density-Based Spatial Clustering of Applications with Noise).
//!
//! This module provides [`DBSCAN`], a density-based clustering algorithm
//! that groups together points that are closely packed together, marking
//! as outliers points that lie alone in low-density regions.
//!
//! # Algorithm
//!
//! 1. For each point, find all neighbors within distance `eps`.
//! 2. Points with at least `min_samples` neighbors (including themselves)
//!    are **core points**.
//! 3. Clusters are formed by connecting core points that are within `eps`
//!    of each other, and assigning border points to the cluster of their
//!    nearest core point.
//! 4. Points that are not reachable from any core point are labeled as
//!    **noise** (label = -1).
//!
//! # Notes
//!
//! DBSCAN does **not** implement [`Predict`](ferrolearn_core::Predict) — it
//! only labels the training data. Use the fitted labels directly.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_cluster::DBSCAN;
//! use ferrolearn_core::Fit;
//! use ndarray::Array2;
//!
//! let x = Array2::from_shape_vec((6, 2), vec![
//!     1.0, 1.0,  1.1, 1.0,  1.0, 1.1,
//!     5.0, 5.0,  5.1, 5.0,  5.0, 5.1,
//! ]).unwrap();
//!
//! let model = DBSCAN::<f64>::new(1.0);
//! let fitted = model.fit(&x, &()).unwrap();
//! let labels = fitted.labels();
//! assert_eq!(labels.len(), 6);
//! ```
//!
//! # `## REQ status`
//!
//! Binary classification (R-DEFER-2): two states only — SHIPPED needs impl + a
//! non-test production consumer + green verification + symbol-anchor + sklearn
//! `file:line`; NOT-STARTED carries the open prereq blocker. **Unlike the
//! `feature_agglomeration.rs` / `spectral.rs` siblings, `DBSCAN` HAS a real PyO3
//! binding** — `#[pyclass(name = "_RsDBSCAN")] struct RsDBSCAN` in
//! `ferrolearn-python/src/extras.rs` (its `fn fit` calls
//! `ferrolearn_cluster::DBSCAN::<f64>::new(self.eps).with_min_samples(self.min_samples)`
//! and `#[getter] labels_` surfaces `f.labels()`), registered via
//! `m.add_class::<extras::RsDBSCAN>()` in `ferrolearn-python/src/lib.rs`, surfaced
//! as `ferrolearn.DBSCAN` (`fit_predict` / `labels_`). So
//! `import ferrolearn; ferrolearn.DBSCAN(...).fit(X).labels_` is the non-test
//! production consumer of the core contract. **Honest assessment (R-HONEST-3):
//! this unit's CORE contract genuinely SHIPS** — DBSCAN is deterministic (no RNG,
//! no iterative optimizer), so on the Euclidean / no-`sample_weight` path
//! `labels_` and `core_sample_indices_` VALUE-match the live sklearn 1.5.2 oracle
//! EXACTLY (element-wise, including the shared-border tie-break and the noise
//! `-1` set), backed by green guards in
//! `ferrolearn-cluster/tests/divergence_dbscan.rs`. The divergence appears ONLY
//! on a neighbor edge where ferrolearn's `sum-of-squares ≤ eps²` and sklearn's
//! `euclidean_distances` (dot-product) / tree distance round to OPPOSITE SIDES of
//! the `eps` boundary — e.g. coords whose true distance is within a ULP of `eps`
//! (the two forms then disagree on inclusion, flipping core status / merging
//! clusters). When the boundary value is exactly representable the two agree
//! (the green guards include Fixture C with edges exactly at `eps=1.0`, where
//! `1.0²` is exact and both include). This exact-boundary distance-form parity is
//! carved out as REQ-11 (NOT-STARTED, #952), tied to the neighbor-search
//! algorithm surface (#949). The NOT-STARTED surface is the
//! unimplemented parameter/attribute surface (`metric`/`p`,
//! `algorithm`/`leaf_size`/`n_jobs`, the `eps=0.5` Rust-constructor default + error
//! ABI, the ferray substrate), each carrying an open
//! prereq blocker. Cites use symbol anchors (ferrolearn) / `file:line`
//! (sklearn 1.5.2, commit 156ef14). Live oracle = installed sklearn 1.5.2. (REQ
//! numbering follows `.design/cluster/dbscan.md`.)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`labels_` VALUE parity, Euclidean / no `sample_weight`) | SHIPPED | impl `Fit::fit` (index-ordered cluster expansion via `VecDeque` BFS) + `fn fit_predict` in `dbscan.rs` mirror `dbscan_inner` (`sklearn/cluster/_dbscan_inner.pyx`) + `_dbscan.py:431-439` (`labels = np.full(n, -1)`, `dbscan_inner(...)`): clusters numbered by the first unlabeled core in index order, noise `= -1`, a border point reachable from two clusters joins the FIRST-reaching cluster (guard `if labels[neighbor] == -1`). VALUE-matches sklearn EXACTLY element-wise on the tested fixtures; diverges only when ferrolearn's `sum-sq ≤ eps²` and sklearn's `euclidean_distances`/tree distance round to opposite sides of the `eps` boundary (true distance within a ULP of `eps`) — carved out as REQ-11 (NOT-STARTED, #952). Non-test consumer: `ferrolearn.DBSCAN.fit(X).labels_` via `RsDBSCAN::labels_` (`ferrolearn-python/src/extras.rs`). Green guards (`tests/divergence_dbscan.rs`): `green_dbscan_labels_two_clusters`, `green_dbscan_labels_three_clusters`, `green_dbscan_shared_border_tie`, `green_dbscan_noise_and_core_indices`. |
//! | REQ-2 (`core_sample_indices_` VALUE parity) | SHIPPED | impl `core_sample_indices = (0..n).filter(\|&i\| is_core[i]).collect()` in `Fit::fit` + accessor `fn core_sample_indices` in `dbscan.rs`, mirroring `core_sample_indices_ = np.where(core_samples)[0]` (`_dbscan.py:438`), core `= neighborhoods[i].len() >= min_samples` (`:435`). VALUE-matches ascending EXACTLY on non-eps-boundary inputs (the eps-boundary case flips a point's core status — REQ-11/#952). Non-test consumer: crate re-export `pub use dbscan::{DBSCAN, FittedDBSCAN}` (`ferrolearn-cluster/src/lib.rs`) — the in-crate accessor (the PyO3 layer surfaces only `labels_`, not `core_sample_indices_`). Green guards: `green_dbscan_shared_border_tie` (Fixture C core `[0,1,2,3,5,6,7,8]`, border idx4 excluded), `green_dbscan_noise_and_core_indices` (Fixture D core `[0,1,2,4,6,7,8,9,13]`). |
//! | REQ-3 (`eps>0` / `min_samples>=1` validation boundary) | SHIPPED | impl `Fit::fit` guards `self.eps <= F::zero()` → `Err(FerroError::InvalidParameter { name: "eps" })` and `self.min_samples == 0` → `Err(FerroError::InvalidParameter { name: "min_samples" })`, matching the accept/reject boundary of `_parameter_constraints` `"eps": [Interval(Real, 0.0, None, closed="neither")]` / `"min_samples": [Interval(Integral, 1, None, closed="left")]` (`_dbscan.py:332-333`). Non-test consumer: `RsDBSCAN::fit` maps the error to `PyValueError` (`ferrolearn-python/src/extras.rs`). Green guards: `green_dbscan_eps_zero_rejected` (`eps=0` → `Err`), `green_dbscan_min_samples_one_self_core` (`min_samples=1` accepted, self-core). GAP: the error TYPE is `FerroError::InvalidParameter`, not sklearn's `InvalidParameterError`/`ValueError` ABI — the boundary itself matches; the ABI nuance is tracked under REQ-4. |
//! | REQ-4 (`eps=0.5` Rust-constructor default + sklearn error ABI) | NOT-STARTED | open prereq blocker **#946**. sklearn `__init__` `eps=0.5` default (`_dbscan.py:347`); ferrolearn `fn new(eps: F)` REQUIRES `eps` — no default. (The PyO3 layer DOES default `eps=0.5` in `RsDBSCAN::new`, but the Rust constructor `DBSCAN::new` does not.) Also validation errors are `FerroError::InvalidParameter`, not the sklearn `InvalidParameterError`/`ValueError` ABI (R-DEV-2). |
//! | REQ-5 (`sample_weight` — alters core determination) | SHIPPED | impl: `DBSCAN<F>` gains `pub sample_weight: Option<Array1<F>>` + builder `fn with_sample_weight` (`dbscan.rs`); `Fit::fit` validates `w.len() == n_samples` (else `FerroError::ShapeMismatch { context: "sample_weight" }`, mirroring `_check_sample_weight`'s shape `ValueError`, `sklearn/utils/validation.py:2055-2060`) then, when `Some(w)`, sets the core mask `is_core[i] = (sum_{j in neighborhoods[i]} w[j]) >= F::from(min_samples)` — a FLOAT compare over the SAME self-including neighborhoods — mirroring `n_neighbors = [sum(sample_weight[neighbors])...]` + `core_samples = n_neighbors >= self.min_samples` (`_dbscan.py:427-435`). `None` keeps the byte-for-byte unchanged `len >= min_samples` path. Downstream BFS / `core_sample_indices_` / `components_` / `labels_` consume the new mask UNCHANGED. Negative/zero weights accepted (DBSCAN does not pass `only_non_negative`). Non-test consumer: crate re-export `pub use dbscan::{DBSCAN, FittedDBSCAN}` (`ferrolearn-cluster/src/lib.rs`) — the in-crate builder/accessors (the PyO3 `sample_weight` Python surface stays a REQ-9 follow-on, matching how `core_sample_indices_`/`components_` already ship through the re-export, not the binding). Live-oracle guards (`tests/divergence_dbscan.rs`, R-CHAR-3): `green_dbscan_weighted_flips_isolated_to_core` (`w=[5,1,1]` → `[0,-1,-1]` core `[0]` vs unweighted `[-1,-1,-1]` core `[]`), `green_dbscan_weighted_demotes_core_to_noise` (`w=[0.5;4]`, `min_samples=4` → all noise vs unweighted one cluster), `green_dbscan_all_ones_equals_unweighted` (`w=[1;8]` reproduces Fixture-A `[0,0,0,0,1,1,1,1]`/core `[0..8]` EXACTLY), `green_dbscan_fractional_weight_boundary` (`w=[0.5;8]`: `8*0.5=4.0 >= 4` core / `< 5` noise — the float `>=` boundary), `green_dbscan_sample_weight_wrong_length_errs` (`Err`, no panic). |
//! | REQ-6 (`metric` / `p` / `metric_params`) | NOT-STARTED | open prereq blocker **#948**. sklearn accepts any `pairwise_distances` metric (`_dbscan.py:334-337`), `p` for Minkowski (`:341`), default `'euclidean'` (`:350`). ferrolearn `fn region_query` / `fn squared_euclidean` (`dbscan.rs`) is Euclidean-ONLY; no `metric`/`p`/`metric_params` param (oracle: `metric='manhattan'` → `[-1,0,0]` vs euclidean `[0,0,0]`). Missing surface. |
//! | REQ-7 (`algorithm` / `leaf_size` / `n_jobs`) | NOT-STARTED | open prereq blocker **#949**. sklearn routes neighbor search through `NearestNeighbors(radius, algorithm, leaf_size, ..., n_jobs)` (`_dbscan.py:411-422`; constraints `:339-342`). ferrolearn uses a fixed brute-force `O(n^2)` `fn region_query` with no parameter. (Brute force value-matches the default `'auto'`; the divergence is the absent parameter surface.) Missing surface. |
//! | REQ-8 (`components_` fitted attribute) | SHIPPED | impl: `Fit::fit` builds `components_: Array2<F>` of shape `(core_sample_indices.len(), n_features)` whose row `k` is `x.row(core_sample_indices[k])`, stored in `FittedDBSCAN`, surfaced via accessor `fn components` in `dbscan.rs` — mirroring `self.components_ = X[self.core_sample_indices_].copy()` (`_dbscan.py:441-446`). VALUE-matches the live sklearn 1.5.2 oracle (`DBSCAN(eps=0.5, min_samples=3)` on the 7-point fixture → `components_` shape `(6,2)` = `[[1,1],[1.2,1.1],[0.9,1.0],[8,8],[8.1,8.2],[8.0,7.9]]`, the rows of `X` at `core_sample_indices_ = [0,1,2,3,4,5]`). Consumer: crate re-export `pub use dbscan::{DBSCAN, FittedDBSCAN}` (`ferrolearn-cluster/src/lib.rs`) — the in-crate accessor (the PyO3 layer does not yet re-expose `components_`, see REQ-9). Green guard: `dbscan_components_match_sklearn` (`dbscan.rs` tests). |
//! | REQ-9 (PyO3 binding VALUE parity) | SHIPPED | impl `#[pyclass(name = "_RsDBSCAN")] RsDBSCAN` (`ferrolearn-python/src/extras.rs`): `fn new(eps=0.5, min_samples=5)`, `fn fit` calling `ferrolearn_cluster::DBSCAN::<f64>::new(self.eps).with_min_samples(self.min_samples)`, `#[getter] labels_`; registered via `m.add_class::<extras::RsDBSCAN>()` (`ferrolearn-python/src/lib.rs`), surfaced as `ferrolearn.DBSCAN` (`fit_predict` / `labels_`). Non-test consumer: `import ferrolearn; ferrolearn.DBSCAN(...).fit(X).labels_`. Since the Rust core value-matches (REQ-1/2), `import ferrolearn` matches `import sklearn` on the Euclidean / no-`sample_weight` path. The binding does NOT yet re-expose `core_sample_indices_` / `components_` / `sample_weight` / `metric` (only `labels_` via `#[getter]`) — those ride their own REQs. |
//! | REQ-10 (ferray substrate) | NOT-STARTED | open prereq blocker **#951**. `dbscan.rs` imports `ndarray::{Array1, Array2}` + `num_traits::Float` + `std::collections::VecDeque`; not migrated to `ferray-core` (R-SUBSTRATE-1/2). The PyO3 boundary uses `numpy2_to_ndarray` (`extras.rs`), an `ndarray` bridge, not `ferray::numpy_interop`. |
//! | REQ-11 (exact eps-boundary neighbor parity) | NOT-STARTED | open prereq blocker **#952**. `fn region_query` (`dbscan.rs`) includes a neighbor when `squared_euclidean ≤ eps*eps`; sklearn routes through `NearestNeighbors(radius=eps).radius_neighbors` (`_dbscan.py:411-422`) whose `euclidean_distances`/tree distance computation rounds differently at the exact boundary. For an edge whose true distance ≈ `eps` (e.g. exactly `1.3` with `eps=1.3`, or `0.9999999999999998` with `eps=1.0`), the two disagree on inclusion → flips core status / merges clusters, diverging from the oracle `labels_`/`core_sample_indices_`. Reproducing sklearn's exact boundary requires its `euclidean_distances` dot-product rounding AND its `algorithm`-dependent neighbor search (#949) — not a single-file fix. Documented (no committed failing test per R-DEFER-6 / the kdtree #831 + balltree #858 convention). |

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::Fit;
use ndarray::{Array1, Array2};
use num_traits::Float;
use std::collections::VecDeque;

/// NumPy-compatible pairwise summation, byte-for-byte mirroring `np.sum` over a
/// 1-D contiguous array (NumPy `pairwise_sum`,
/// `numpy/_core/src/umath/loops_utils.h.src:81-145`).
///
/// DBSCAN's weighted core determination compares `np.sum(sample_weight[neighbors])`
/// against `min_samples` (`sklearn/cluster/_dbscan.py:428-435`). A sequential
/// left-fold rounds differently from NumPy's pairwise reduction (e.g. ten `0.1`
/// weights sum to exactly `1.0` under pairwise but `0.999…` sequentially),
/// flipping the float `>=` boundary, so the weight sum must use this exact
/// reduction order (#2189).
fn numpy_pairwise_sum<F: Float>(a: &[F]) -> F {
    let n = a.len();
    if n < 8 {
        // Start from -0.0 to preserve `-0` semantics (numpy `:86-89`).
        let mut res = F::neg_zero();
        for &v in a {
            res = res + v;
        }
        res
    } else if n <= 128 {
        // Eight-accumulator unrolled block (numpy `:96-135`).
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let limit = n - (n % 8);
        let mut i = 8;
        while i < limit {
            r[0] = r[0] + a[i];
            r[1] = r[1] + a[i + 1];
            r[2] = r[2] + a[i + 2];
            r[3] = r[3] + a[i + 3];
            r[4] = r[4] + a[i + 4];
            r[5] = r[5] + a[i + 5];
            r[6] = r[6] + a[i + 6];
            r[7] = r[7] + a[i + 7];
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        // Non-multiple-of-8 remainder (numpy `:131-134`).
        while i < n {
            res = res + a[i];
            i += 1;
        }
        res
    } else {
        // Divide and conquer, n2 rounded down to a multiple of 8 (numpy `:137-143`).
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        numpy_pairwise_sum(&a[..n2]) + numpy_pairwise_sum(&a[n2..])
    }
}

/// DBSCAN clustering configuration (unfitted).
///
/// Holds hyperparameters for the DBSCAN algorithm. Call [`Fit::fit`]
/// to run the algorithm and produce a [`FittedDBSCAN`].
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct DBSCAN<F> {
    /// The maximum distance between two samples for them to be considered
    /// as in the same neighborhood.
    pub eps: F,
    /// The minimum number of samples in a neighborhood for a point to be
    /// considered a core point (including the point itself).
    pub min_samples: usize,
    /// Optional per-sample weights. When `Some(w)`, a point `i` is a core
    /// point iff the SUM of weights over its neighbors (including itself)
    /// is `>= min_samples`, mirroring sklearn's
    /// `n_neighbors[i] = sum(sample_weight[neighbors])`
    /// (`sklearn/cluster/_dbscan.py:427-429`). `None` is equivalent to all
    /// ones (the unweighted `len >= min_samples` path).
    pub sample_weight: Option<Array1<F>>,
}

impl<F: Float> DBSCAN<F> {
    /// Create a new `DBSCAN` with the given `eps` radius.
    ///
    /// Uses default `min_samples = 5` and no sample weights.
    #[must_use]
    pub fn new(eps: F) -> Self {
        Self {
            eps,
            min_samples: 5,
            sample_weight: None,
        }
    }

    /// Set the minimum number of samples for core points.
    #[must_use]
    pub fn with_min_samples(mut self, min_samples: usize) -> Self {
        self.min_samples = min_samples;
        self
    }

    /// Set per-sample weights.
    ///
    /// Mirrors scikit-learn's `DBSCAN.fit(X, sample_weight=w)`: a sample with
    /// a weight of at least `min_samples` is by itself a core sample, and a
    /// sample with a negative weight may inhibit its eps-neighbors from being
    /// core (`sklearn/cluster/_dbscan.py:384-388`). Weights are absolute and
    /// default to 1 (passing all-ones reproduces the unweighted path exactly).
    /// Negative and zero weights are accepted (sklearn's `_check_sample_weight`
    /// does NOT require non-negativity for DBSCAN). The weight vector length is
    /// validated against the number of samples at [`Fit::fit`] time.
    #[must_use]
    pub fn with_sample_weight(mut self, w: Array1<F>) -> Self {
        self.sample_weight = Some(w);
        self
    }
}

/// Fitted DBSCAN model.
///
/// Stores the cluster labels and core sample indices from the training run.
/// Noise points are labeled with -1.
///
/// DBSCAN does **not** implement [`Predict`](ferrolearn_core::Predict).
#[derive(Debug, Clone)]
pub struct FittedDBSCAN<F> {
    /// Cluster label for each training sample. Noise points have label -1.
    labels_: Array1<isize>,
    /// Indices of core samples in the training data.
    core_sample_indices_: Vec<usize>,
    /// Coordinates of the core samples (rows of `X` at `core_sample_indices_`).
    pub(crate) components_: Array2<F>,
    /// Phantom data to retain the float type parameter.
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float> FittedDBSCAN<F> {
    /// Return the cluster labels for the training data.
    ///
    /// Noise points have label `-1`.
    #[must_use]
    pub fn labels(&self) -> &Array1<isize> {
        &self.labels_
    }

    /// Return the indices of core samples in the training data.
    #[must_use]
    pub fn core_sample_indices(&self) -> &[usize] {
        &self.core_sample_indices_
    }

    /// Return the coordinates of the core samples, of shape
    /// `(n_core_samples, n_features)`.
    ///
    /// Row `k` is the training row `X[core_sample_indices()[k]]`, mirroring
    /// sklearn's `self.components_ = X[self.core_sample_indices_].copy()`
    /// (`sklearn/cluster/_dbscan.py:441-446`).
    #[must_use]
    pub fn components(&self) -> &Array2<F> {
        &self.components_
    }

    /// Return the number of clusters found (excluding noise).
    #[must_use]
    pub fn n_clusters(&self) -> usize {
        let max_label = self.labels_.iter().max().copied().unwrap_or(-1);
        if max_label < 0 {
            0
        } else {
            (max_label + 1) as usize
        }
    }
}

/// Compute the squared Euclidean distance between two slices.
fn squared_euclidean<F: Float>(a: &[F], b: &[F]) -> F {
    a.iter()
        .zip(b.iter())
        .fold(F::zero(), |acc, (&ai, &bi)| acc + (ai - bi) * (ai - bi))
}

/// Find all neighbors within `eps` distance of point `idx`.
fn region_query<F: Float>(x: &Array2<F>, idx: usize, eps_sq: F) -> Vec<usize> {
    let n_samples = x.nrows();
    let row = x.row(idx);
    let row_slice = row.as_slice().unwrap_or(&[]);

    let mut neighbors = Vec::new();
    for j in 0..n_samples {
        let other = x.row(j);
        let other_slice = other.as_slice().unwrap_or(&[]);
        if squared_euclidean(row_slice, other_slice) <= eps_sq {
            neighbors.push(j);
        }
    }
    neighbors
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for DBSCAN<F> {
    type Fitted = FittedDBSCAN<F>;
    type Error = FerroError;

    /// Fit the DBSCAN model to the data.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if `eps` is not positive
    /// or `min_samples` is zero.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedDBSCAN<F>, FerroError> {
        // Validate parameters.
        if self.eps <= F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "eps".into(),
                reason: "must be positive".into(),
            });
        }

        if self.min_samples == 0 {
            return Err(FerroError::InvalidParameter {
                name: "min_samples".into(),
                reason: "must be at least 1".into(),
            });
        }

        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Validate `sample_weight` length against `n_samples`, mirroring
        // `_check_sample_weight` (`sklearn/utils/validation.py:2055-2060`):
        // a wrong-length weight vector raises `ValueError`. Negative/zero
        // weights are ALLOWED (DBSCAN does not pass `only_non_negative`).
        if let Some(w) = &self.sample_weight
            && w.len() != n_samples
        {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![w.len()],
                context: "sample_weight".into(),
            });
        }

        if n_samples == 0 {
            return Ok(FittedDBSCAN {
                labels_: Array1::zeros(0),
                core_sample_indices_: Vec::new(),
                components_: Array2::zeros((0, n_features)),
                _marker: std::marker::PhantomData,
            });
        }

        let eps_sq = self.eps * self.eps;

        // Step 1: Find neighborhoods for all points.
        let neighborhoods: Vec<Vec<usize>> =
            (0..n_samples).map(|i| region_query(x, i, eps_sq)).collect();

        // Step 2: Identify core points.
        //
        // Unweighted (`sample_weight` is `None`): `is_core[i]` iff the
        // neighborhood size (self included) is `>= min_samples`, mirroring
        // `n_neighbors = [len(neighbors) ...]` (`_dbscan.py:425`).
        //
        // Weighted: `is_core[i]` iff the SUM of weights over the SAME
        // neighborhood (self included) is `>= min_samples` (a float compare),
        // mirroring `n_neighbors = [sum(sample_weight[neighbors]) ...]` and
        // `core_samples = n_neighbors >= self.min_samples`
        // (`_dbscan.py:427-435`). `min_samples` is cast to `F` for the compare.
        let is_core: Vec<bool> = match &self.sample_weight {
            None => neighborhoods
                .iter()
                .map(|n| n.len() >= self.min_samples)
                .collect(),
            Some(w) => {
                let min_samples_f = F::from(self.min_samples).unwrap_or_else(F::infinity);
                neighborhoods
                    .iter()
                    .map(|n| {
                        // sklearn computes `np.sum(sample_weight[neighbors])`
                        // (`_dbscan.py:428`); NumPy `np.sum` uses PAIRWISE
                        // summation, not a sequential fold, so a running sum that
                        // straddles `min_samples` (e.g. ten `0.1` weights ->
                        // numpy 1.0 vs sequential 0.999…) must use the SAME
                        // reduction order to match the float `>=` core boundary
                        // (#2189).
                        let neighbor_weights: Vec<F> = n.iter().map(|&j| w[j]).collect();
                        let weight_sum = numpy_pairwise_sum(&neighbor_weights);
                        weight_sum >= min_samples_f
                    })
                    .collect()
            }
        };

        let core_sample_indices: Vec<usize> = (0..n_samples).filter(|&i| is_core[i]).collect();

        // Retain the coordinates of the core samples, mirroring sklearn's
        // `self.components_ = X[self.core_sample_indices_].copy()`
        // (`sklearn/cluster/_dbscan.py:441-446`).
        let mut components_ = Array2::<F>::zeros((core_sample_indices.len(), n_features));
        for (k, &idx) in core_sample_indices.iter().enumerate() {
            components_.row_mut(k).assign(&x.row(idx));
        }

        // Step 3: Expand clusters from core points via BFS.
        let mut labels = Array1::from_elem(n_samples, -1isize);
        let mut current_cluster: isize = -1;

        for i in 0..n_samples {
            // Skip non-core or already-labeled points.
            if !is_core[i] || labels[i] != -1 {
                continue;
            }

            // Start a new cluster.
            current_cluster += 1;
            labels[i] = current_cluster;

            // BFS expansion.
            let mut queue: VecDeque<usize> = VecDeque::new();
            for &neighbor in &neighborhoods[i] {
                if neighbor != i {
                    queue.push_back(neighbor);
                }
            }

            while let Some(q) = queue.pop_front() {
                if labels[q] == -1 {
                    // Assign to current cluster (was noise or unvisited).
                    labels[q] = current_cluster;
                }

                if !is_core[q] {
                    // Border point — don't expand further.
                    continue;
                }

                // If this core point was already assigned to this cluster
                // by a prior BFS step, skip expanding again.
                if labels[q] == current_cluster {
                    // Expand: add unvisited neighbors to queue.
                    for &neighbor in &neighborhoods[q] {
                        if labels[neighbor] == -1 {
                            labels[neighbor] = current_cluster;
                            if is_core[neighbor] {
                                queue.push_back(neighbor);
                            }
                        }
                    }
                }
            }
        }

        Ok(FittedDBSCAN {
            labels_: labels,
            core_sample_indices_: core_sample_indices,
            components_,
            _marker: std::marker::PhantomData,
        })
    }
}

impl<F: Float + Send + Sync + 'static> DBSCAN<F> {
    /// Fit on `x` and return the cluster labels for those samples in one
    /// call. Equivalent to sklearn `ClusterMixin.fit_predict`. Noise
    /// samples are labeled as `-1`.
    ///
    /// # Errors
    ///
    /// Forwards any error from [`Fit::fit`].
    pub fn fit_predict(&self, x: &Array2<F>) -> Result<Array1<isize>, FerroError> {
        let fitted = self.fit(x, &())?;
        Ok(fitted.labels().clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two well-separated clusters.
    fn make_two_clusters() -> Array2<f64> {
        Array2::from_shape_vec(
            (8, 2),
            vec![
                // Cluster A near (0, 0)
                0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, // Cluster B near (10, 10)
                10.0, 10.0, 10.5, 10.0, 10.0, 10.5, 10.5, 10.5,
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_simple_clusters() {
        let x = make_two_clusters();
        let model = DBSCAN::<f64>::new(1.5).with_min_samples(2);
        let fitted = model.fit(&x, &()).unwrap();

        let labels = fitted.labels();
        assert_eq!(labels.len(), 8);

        // First 4 should be in one cluster, last 4 in another.
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[0], labels[3]);
        assert_eq!(labels[4], labels[5]);
        assert_eq!(labels[4], labels[6]);
        assert_eq!(labels[4], labels[7]);
        assert_ne!(labels[0], labels[4]);

        // Two clusters found.
        assert_eq!(fitted.n_clusters(), 2);
    }

    #[test]
    fn test_noise_detection() {
        // Two tight clusters + one outlier.
        let x = Array2::from_shape_vec(
            (7, 2),
            vec![
                0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1, // Outlier
                100.0, 100.0,
            ],
        )
        .unwrap();

        let model = DBSCAN::<f64>::new(0.5).with_min_samples(2);
        let fitted = model.fit(&x, &()).unwrap();

        let labels = fitted.labels();
        // Outlier should be noise.
        assert_eq!(labels[6], -1);
        // Others should not be noise.
        assert!(labels[0] >= 0);
        assert!(labels[3] >= 0);
    }

    #[test]
    fn test_core_border_noise_identification() {
        // Ring of points: 3 core, 1 border, 1 noise.
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![
                0.0, 0.0, // core (neighbors: 0,1,2)
                0.3, 0.0, // core (neighbors: 0,1,2)
                0.0, 0.3, // core (neighbors: 0,1,2)
                0.6, 0.0, // border (neighbor: 1 at least)
                10.0, 10.0, // noise
            ],
        )
        .unwrap();

        let model = DBSCAN::<f64>::new(0.5).with_min_samples(3);
        let fitted = model.fit(&x, &()).unwrap();

        let labels = fitted.labels();
        let core_indices = fitted.core_sample_indices();

        // Points 0, 1, 2 should be core points.
        assert!(core_indices.contains(&0));
        assert!(core_indices.contains(&1));
        assert!(core_indices.contains(&2));

        // Point 3 is a border point (reachable from core point 1).
        assert!(labels[3] >= 0);
        assert!(!core_indices.contains(&3));

        // Point 4 is noise.
        assert_eq!(labels[4], -1);
    }

    #[test]
    fn test_all_noise_with_high_min_samples() {
        let x =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();

        // min_samples too high: no point has enough neighbors.
        let model = DBSCAN::<f64>::new(0.5).with_min_samples(100);
        let fitted = model.fit(&x, &()).unwrap();

        let labels = fitted.labels();
        for &label in labels {
            assert_eq!(label, -1);
        }
        assert_eq!(fitted.n_clusters(), 0);
    }

    #[test]
    fn test_all_noise_with_tiny_eps() {
        let x =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();

        // eps too small: no points are neighbors of each other (except self).
        let model = DBSCAN::<f64>::new(0.001).with_min_samples(2);
        let fitted = model.fit(&x, &()).unwrap();

        for &label in fitted.labels() {
            assert_eq!(label, -1);
        }
    }

    #[test]
    fn test_single_point_cluster() {
        // One cluster + one isolated point.
        let x = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 100.0, 100.0])
            .unwrap();

        let model = DBSCAN::<f64>::new(0.5).with_min_samples(2);
        let fitted = model.fit(&x, &()).unwrap();

        // Isolated point should be noise.
        assert_eq!(fitted.labels()[3], -1);
        // Others should be in a cluster.
        assert!(fitted.labels()[0] >= 0);
    }

    #[test]
    fn test_all_in_one_cluster() {
        let x =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1]).unwrap();

        let model = DBSCAN::<f64>::new(1.0).with_min_samples(2);
        let fitted = model.fit(&x, &()).unwrap();

        // All should be in the same cluster.
        let labels = fitted.labels();
        assert!(labels[0] >= 0);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[0], labels[3]);
        assert_eq!(fitted.n_clusters(), 1);
    }

    #[test]
    fn test_empty_data() {
        let x = Array2::<f64>::zeros((0, 2));
        let model = DBSCAN::<f64>::new(1.0);
        let fitted = model.fit(&x, &()).unwrap();

        assert_eq!(fitted.labels().len(), 0);
        assert!(fitted.core_sample_indices().is_empty());
        assert_eq!(fitted.n_clusters(), 0);
        // components_ is (0, n_features) on empty input, matching sklearn's
        // `np.empty((0, X.shape[1]))` (_dbscan.py:446).
        assert_eq!(fitted.components().shape(), &[0, 2]);
    }

    #[test]
    fn test_single_sample() {
        let x = Array2::from_shape_vec((1, 2), vec![5.0, 5.0]).unwrap();
        let model = DBSCAN::<f64>::new(1.0).with_min_samples(1);
        let fitted = model.fit(&x, &()).unwrap();

        // With min_samples=1, a single point is a core point and its own cluster.
        assert_eq!(fitted.labels()[0], 0);
        assert_eq!(fitted.core_sample_indices(), &[0]);
    }

    #[test]
    fn test_single_sample_noise() {
        let x = Array2::from_shape_vec((1, 2), vec![5.0, 5.0]).unwrap();
        let model = DBSCAN::<f64>::new(1.0).with_min_samples(2);
        let fitted = model.fit(&x, &()).unwrap();

        // With min_samples=2, a single point cannot be a core point.
        assert_eq!(fitted.labels()[0], -1);
    }

    #[test]
    fn test_invalid_eps() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]).unwrap();
        let model = DBSCAN::<f64>::new(-1.0);
        let result = model.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_eps() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]).unwrap();
        let model = DBSCAN::<f64>::new(0.0);
        let result = model.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_min_samples() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]).unwrap();
        let model = DBSCAN::<f64>::new(1.0).with_min_samples(0);
        let result = model.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_core_sample_indices_correct() {
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1, 100.0, 100.0],
        )
        .unwrap();

        let model = DBSCAN::<f64>::new(0.5).with_min_samples(3);
        let fitted = model.fit(&x, &()).unwrap();

        // All core samples should have enough neighbors.
        for &idx in fitted.core_sample_indices() {
            assert!(idx < 5);
        }
    }

    #[test]
    fn test_f32_support() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                0.0f32, 0.0, 0.1, 0.0, 0.0, 0.1, 10.0, 10.0, 10.1, 10.0, 10.0, 10.1,
            ],
        )
        .unwrap();

        let model = DBSCAN::<f32>::new(0.5).with_min_samples(2);
        let fitted = model.fit(&x, &()).unwrap();

        assert_eq!(fitted.labels().len(), 6);
        assert_eq!(fitted.n_clusters(), 2);
    }

    #[test]
    fn test_three_clusters() {
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1, 10.0, 0.0, 10.1, 0.0,
                10.0, 0.1,
            ],
        )
        .unwrap();

        let model = DBSCAN::<f64>::new(0.5).with_min_samples(2);
        let fitted = model.fit(&x, &()).unwrap();

        assert_eq!(fitted.n_clusters(), 3);

        let labels = fitted.labels();
        // Same cluster groupings.
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[3], labels[5]);
        assert_eq!(labels[6], labels[7]);
        assert_eq!(labels[6], labels[8]);
        // Different clusters.
        assert_ne!(labels[0], labels[3]);
        assert_ne!(labels[0], labels[6]);
        assert_ne!(labels[3], labels[6]);
    }

    #[test]
    fn dbscan_components_match_sklearn() {
        // Live sklearn 1.5.2 oracle (run from /tmp):
        //   X = [[1,1],[1.2,1.1],[0.9,1.0],[8,8],[8.1,8.2],[8.0,7.9],[5,5]]
        //   DBSCAN(eps=0.5, min_samples=3).fit(X)
        //   labels_              = [0,0,0,1,1,1,-1]
        //   core_sample_indices_ = [0,1,2,3,4,5]
        //   components_          = [[1,1],[1.2,1.1],[0.9,1.0],[8,8],[8.1,8.2],[8.0,7.9]]
        // Mirrors `self.components_ = X[self.core_sample_indices_].copy()`
        // (sklearn/cluster/_dbscan.py:441-446).
        let x = Array2::from_shape_vec(
            (7, 2),
            vec![
                1.0, 1.0, 1.2, 1.1, 0.9, 1.0, 8.0, 8.0, 8.1, 8.2, 8.0, 7.9, 5.0, 5.0,
            ],
        )
        .unwrap();

        let model = DBSCAN::<f64>::new(0.5).with_min_samples(3);
        let fitted = model.fit(&x, &()).unwrap();

        // core_sample_indices_ oracle.
        assert_eq!(fitted.core_sample_indices(), &[0, 1, 2, 3, 4, 5]);

        let oracle = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 1.2, 1.1, 0.9, 1.0, 8.0, 8.0, 8.1, 8.2, 8.0, 7.9],
        )
        .unwrap();

        let components = fitted.components();
        assert_eq!(components.shape(), &[6, 2]);

        // Element-wise match against the oracle matrix (<= 1e-12).
        for (got, want) in components.iter().zip(oracle.iter()) {
            assert!(
                (got - want).abs() <= 1e-12,
                "components_ mismatch: got {got}, want {want}"
            );
        }

        // Invariant: each row of components_ equals X[core_sample_indices()[k]].
        for (k, &idx) in fitted.core_sample_indices().iter().enumerate() {
            for j in 0..x.ncols() {
                assert!(
                    (components[[k, j]] - x[[idx, j]]).abs() <= 1e-12,
                    "row {k} != X[{idx}] at col {j}"
                );
            }
        }
    }

    #[test]
    fn test_identical_points() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).unwrap();

        let model = DBSCAN::<f64>::new(0.5).with_min_samples(2);
        let fitted = model.fit(&x, &()).unwrap();

        // All identical points should be in the same cluster.
        let labels = fitted.labels();
        assert!(labels[0] >= 0);
        for &label in labels {
            assert_eq!(label, labels[0]);
        }
    }
}
