//! OPTICS — Ordering Points To Identify the Clustering Structure.
//!
//! This module provides [`OPTICS`], a density-based algorithm that computes a
//! **reachability ordering** of the data.  Unlike DBSCAN, OPTICS does not
//! require a global density threshold; instead it produces a reachability plot
//! from which clusters at various density levels can be extracted.
//!
//! # Algorithm
//!
//! 1. For each unprocessed point `p`:
//!    - Compute its **core distance** — the distance to the `min_samples`-th
//!      nearest neighbour (within `max_eps`), or `∞` if there are fewer than
//!      `min_samples` neighbours.
//!    - If `p` is a core point, update the reachability distances of all
//!      unprocessed neighbours and add them to an ordered seed list.
//!    - Append `p` to the ordering with its final reachability distance.
//!
//! 2. **Cluster extraction** via the Xi method (see [`FittedOPTICS::extract_clusters`]):
//!    steep descents in the reachability plot define cluster boundaries.
//!
//! OPTICS does **not** implement [`Predict`](ferrolearn_core::Predict) — it
//! produces a reachability ordering and reachability distances from which
//! cluster memberships can be derived post-hoc.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_cluster::OPTICS;
//! use ferrolearn_core::Fit;
//! use ndarray::Array2;
//!
//! let x = Array2::from_shape_vec((9, 2), vec![
//!     0.0, 0.0,  0.1, 0.1,  0.0, 0.1,
//!     5.0, 5.0,  5.1, 5.0,  5.0, 5.1,
//!     10.0, 0.0, 10.1, 0.0, 10.0, 0.1,
//! ]).unwrap();
//!
//! let fitted = OPTICS::<f64>::new(2).fit(&x, &()).unwrap();
//! assert_eq!(fitted.ordering().len(), 9);
//! ```
//!
//! # `## REQ status`
//!
//! Binary (R-DEFER-2), translating `sklearn/cluster/_optics.py` (`class OPTICS`,
//! `compute_optics_graph`, `cluster_optics_xi`, `cluster_optics_dbscan`). Design
//! doc: `.design/cluster/optics.md`. Cites use ferrolearn symbol anchors / sklearn
//! `file:line` (commit 156ef14); expected values from the live sklearn 1.5.2 oracle
//! (R-CHAR-3). OPTICS is deterministic (no RNG) → value-parity is genuine. No PyO3
//! binding — only consumer is the crate re-export. After fixing the traversal
//! (#1080), `core_distances_`/`ordering_`/`reachability_` VALUE-match sklearn, and
//! `predecessor_` (the `-1`-sentinel int array, REQ-4) now value-matches too. The
//! `cluster_method='dbscan'` + `eps` extraction (REQ-6) is now SHIPPED — `fn
//! cluster_optics_dbscan` value-matches the live oracle. The Xi `labels_`
//! (REQ-5), the in-Xi `min_cluster_size` criterion 3.a (REQ-7), and the
//! `cluster_hierarchy_` attribute (REQ-8) are now SHIPPED: `fn cluster_optics_xi`
//! / `fn xi_cluster` faithfully port sklearn's `cluster_optics_xi` / `_xi_cluster`
//! (incl. the paper-corrected steep inequalities, `predecessor_correction`, the
//! in-loop size cut, and `_extract_xi_labels`), bit-exact vs the live oracle. The
//! remaining parameter/attribute surface (REQ-9 toggle, REQ-10 param surface,
//! REQ-12 PyO3, REQ-13 ferray) stays divergent.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`core_distances_` VALUE) | SHIPPED | impl `fn core_distance` (distance to the `(min_samples-1)`-th other point within `max_eps`) value-matches sklearn `_compute_core_distances_` = `kneighbors(X,min_samples)[0][:,-1]` (`_optics.py:405-438`) even on hard fixtures. Consumer: crate re-export `pub use optics::{FittedOPTICS, OPTICS}` (`lib.rs`). Guards: `green_core_distances_three_blobs`, `green_core_distances_small10_ms2/ms3` in `tests/divergence_optics.rs` (live-oracle). |
//! | REQ-2 (`ordering_` traversal value-parity) | SHIPPED | impl public `compute_optics_graph` and `Fit::fit` now select the next seed by LINEAR ARGMIN over all unprocessed reachability with smallest-index tie-break (single pool, no heap), matching sklearn `compute_optics_graph` (`_optics.py:638-659`, `:53` "we do not employ a heap"). The prior `BinaryHeap` traversal diverged on tie-prone data. Guard: `green_compute_optics_graph_public_helper_docstring`, `green_ordering_small10` (sklearn `[0,1,3,9,5,6,7,8,2,4]`; was `[…5,7,8,6…]`) + `green_ordering_three_blobs`/`green_ordering_docstring`. Fixed #1080. |
//! | REQ-3 (`reachability_` VALUE) | SHIPPED | impl `fn update_seeds` computes `max(core_dist_p, dist)` then rounds via `fn round_to_precision` (`np.around(decimals=np.finfo(dtype).precision)`, round-ties-even = numpy rint, `_optics.py:711`); combined with the REQ-2 traversal the reported plot value-matches sklearn (the #1080 ordering parity on tie fixtures REQUIRES the matching rounded reachability at each argmin step). Guards: `green_reachability_docstring` + (noisy) the ordering parity which is driven by reachability. Fixed #1080 (bundled). |
//! | REQ-4 (`predecessor_` `-1`-sentinel int array) | SHIPPED | impl: `FittedOPTICS` stores `predecessor_: Array1<i64>` built in `Fit::fit` (`predecessors.iter().map(\|p\| p.map_or(-1, \|j\| j as i64))`), surfaced via accessor `fn predecessor() -> &Array1<i64>`. This is the `-1`-sentinel int array (shape `(n_samples,)`, indexed by original sample index) matching sklearn `predecessor_` "Seed points have a predecessor of -1" (`_optics.py:187-189`, `np.full(n_samples,-1,dtype=int)` `:604-605`, set on STRICT improvement `:712-714`). ferrolearn's `fn update_seeds` records the predecessor under the SAME strict `new_reach < reachability[q]` condition as sklearn's `improved = np.where(rdists < ...)` (`:712`), so the VALUES match too — verified element-wise (incl. the `-1` seed) vs the live oracle on `three_blobs` (`[-1,0,0,1,3,3,8,6,4]`), `small10` (tie-prone, `[-1,0,3,1,8,9,5,6,7,3]`), and the docstring fixture (`[-1,0,1,5,3,2]`). Internal `fn predecessors() -> &[Option<usize>]` kept (seed `None`) for the Xi extractor + back-compat. Consumer: crate re-export `pub use optics::{FittedOPTICS, OPTICS}` (`lib.rs`). Guards (live-oracle, R-CHAR-3): `green_predecessor_three_blobs`, `green_predecessor_small10`, `green_predecessor_docstring` in `tests/divergence_optics.rs`. **f32 caveat (#2195):** sklearn OPTICS ALWAYS computes the graph in float64 — `reachability_.dtype == float64` even for float32 input (it upcasts) — whereas ferrolearn computes generically in `F`, so on f32 a near-tie in the reachability plot can round to the opposite side, swapping `ordering_`/`reachability_` and hence `predecessor_` (e.g. `small10` f32 at indices 6/7/8). This is an OPTICS-WIDE f32 divergence (it equally affects the SHIPPED f64-only REQ-1/2/3), tracked as #2195 and pinned `#[ignore]` in `tests/divergence_optics_predecessor_f32.rs`; the f64 path is bit-exact. The future fix is to compute the OPTICS graph in f64 regardless of `F` (matching sklearn's upcast) — out of this single-file predecessor scope. |
//! | REQ-5 (`labels_` Xi VALUE parity) | SHIPPED | impl: `Fit::fit`'s `OpticsClusterMethod::Xi` arm calls `fn cluster_optics_xi` → `fn xi_cluster` → `fn extract_xi_labels`, a faithful port of sklearn `cluster_optics_xi`/`_xi_cluster`/`_extract_xi_labels` (`sklearn/cluster/_optics.py:810-1201`): the appended `+inf` sentinel (`:1070`), `xi_complement = 1-xi`, the IEEE-754 `ratio` (Rust `inf/inf=NaN`, `x/inf=0`, `x/0=+inf` reproduce numpy `errstate(invalid="ignore")` `:1082-1087`), the PAPER-CORRECTED steep inequalities (`steep_downward = ratio >= 1/xi_complement` `:1085`, `steep_upward = ratio <= xi_complement` `:1084`), the steep-down/steep-up branches with `fn extend_region`/`fn update_filter_sdas`/`fn correct_predecessor`, Definition-11 criteria 1-4 (incl. the `:1143` `r(x) > D_max` paper-correction), and `_extract_xi_labels`' leaf-only assign + `labels[ordering]=labels` scatter (`:1194-1200`). Consumer (R-DEFER-1, non-test): the `Xi` arm of `Fit::fit` populates `labels_`, reachable via the crate re-export `pub use optics::{FittedOPTICS, OPTICS}` (`lib.rs`); also surfaced through `FittedOPTICS::extract_clusters`. Guards (live-oracle, R-CHAR-3): `green_xi_labels_hierarchy_doctest` (the `_optics.py:891-896` doctest `[0,0,0,1,1,1]`), `green_xi_labels_hierarchy_three_blobs`, `green_xi_labels_hierarchy_small10`, `green_xi_min_cluster_size_inside_loop`, `green_xi_varied_threshold_small10` in `tests/divergence_optics.rs`. **f32 caveat:** the underlying graph is f32-divergent (#2195); these guards are f64. |
//! | REQ-6 (`cluster_method='dbscan'` + `eps` + `cluster_optics_dbscan`) | SHIPPED | impl: free fn `cluster_optics_dbscan(reachability, core_distances, ordering, eps) -> Array1<isize>` translates sklearn's two-step labelling EXACTLY (`sklearn/cluster/_optics.py:781-787`): step 1 walks `ordering` accumulating `cumsum(far_reach & near_core) - 1` with `far_reach = reachability > eps` (STRICT) and `near_core = core_distances <= eps` (INCLUSIVE); step 2 overwrites `labels[far_reach & !near_core] = -1`. Wired into `Fit::fit` via the new `cluster_method: OpticsClusterMethod` field (`enum {Xi (default), Dbscan}`, builder `fn with_cluster_method`) + `eps: Option<F>` field (builder `fn with_eps`): the `Dbscan` arm resolves `eps = self.eps.unwrap_or(self.max_eps)` and errors (`FerroError::InvalidParameter`) when `eps > self.max_eps`, mirroring sklearn's `eps` resolution + ValueError (`:375-383`). Consumer (R-DEFER-1, non-test): `Fit::fit`'s `OpticsClusterMethod::Dbscan` arm CALLS `cluster_optics_dbscan` to populate `labels_`; reachable publicly via `pub mod optics` (`ferrolearn_cluster::optics::{cluster_optics_dbscan, OpticsClusterMethod}`). Guards (live-oracle, R-CHAR-3): `green_dbscan_three_blobs_eps05/eps_none_all_noise/eps_large_merge`, `green_dbscan_small10_eps07_mixed/eps2_merge/eps05_noise`, `green_dbscan_eps_gt_max_eps_errs`, `green_xi_default_unchanged_three_blobs` in `tests/divergence_optics.rs`. (The Xi `labels_` REQ-5/7/8 + the PyO3 surface REQ-12 stay separate NOT-STARTED.) **eps-at-core-distance-boundary caveat (#2196):** when `eps` is set EXACTLY equal to one of ferrolearn's `core_distances_` values, the `<=` near-core cut can flip vs sklearn because ferrolearn's `core_distance` (`sqrt(sum-of-squares)`) differs from sklearn's `kneighbors`/`euclidean_distances` distance form by a sub-ULP (~4e-17), so `core <= eps` rounds to opposite sides at the exact boundary (`small10` at `eps == core_distances_[5]`). This is the SAME distance-form boundary class as DBSCAN #952 (REQ-1 `core_distances_` value-matches to ~1e-9, not bit-exact); for any `eps` not landing precisely on a differing core-distance the cut is value-exact. Tracked #2196, pinned `#[ignore]` in `tests/divergence_optics_dbscan_boundary.rs`. |
//! | REQ-7 (Xi `min_cluster_size` criterion 3.a) | SHIPPED | impl: `fn xi_cluster` applies the size cut INSIDE the steep-up loop — `if c_end + 1 - c_start < min_cluster_size { continue; }` — matching sklearn's `if c_end - c_start + 1 < min_cluster_size: continue` (`sklearn/cluster/_optics.py:1154-1156`), NOT a post-filter (the old `fn filter_small_clusters` post-pass was REMOVED). `min_cluster_size` resolves `None → min_samples` in `fn cluster_optics_xi` (`:902-903`). This is a DIFFERENT fixed point from a post-filter: dropping a cluster mid-loop changes which later intervals get labelled AND the hierarchy. Consumer: same `Fit::fit` Xi arm as REQ-5. Guard (live-oracle): `green_xi_min_cluster_size_inside_loop` — `blobs44` with `min_samples=3`: the 4-point small blob keeps a label at the default `min_cluster_size` but is filtered at `min_cluster_size=10`, changing BOTH `labels_` AND `cluster_hierarchy_` exactly as sklearn. |
//! | REQ-8 (`cluster_hierarchy_` attribute) | SHIPPED | impl: `FittedOPTICS` stores `cluster_hierarchy_: Array2<i64>` (`(n_clusters, 2)` `[start, end]` plot-order intervals, both inclusive), surfaced via `fn cluster_hierarchy() -> &Array2<i64>`. Built in `fn cluster_optics_xi` from the `Vec<(usize,usize)>` `fn xi_cluster` returns (`np.array(clusters)`, `sklearn/cluster/_optics.py:1172`), set on the `Xi` arm of `Fit::fit` (`self.cluster_hierarchy_ = clusters_`, `:373`); the `Dbscan` arm leaves it empty (shape `(0,2)`) — sklearn only sets it on the Xi branch. The order is `_xi_cluster`'s discovery order (U_clusters reversed), which equals the docstring's `(end, -start)` ascending on these fixtures (verified vs the oracle; `cluster_optics_xi` does NOT re-sort — `cluster_hierarchy_ = clusters_` directly). Consumer (R-DEFER-1, non-test): the `Xi` arm of `Fit::fit` populates the field; reachable via the crate re-export. Guards (live-oracle): `green_xi_labels_hierarchy_doctest` (the `_optics.py:893-896` doctest `[[0,2],[3,5],[0,5]]`, with `len(hierarchy)=3 > 2=n_unique_labels` asserting the nested-cluster-kept contract), `_three_blobs`, `_small10`, `green_xi_min_cluster_size_inside_loop`, `green_xi_varied_threshold_small10`, `green_dbscan_cluster_hierarchy_empty` in `tests/divergence_optics.rs`. |
//! | REQ-9 (`predecessor_correction` toggle) | SHIPPED | impl: `OPTICS<F>` carries `predecessor_correction: bool` (default `true` in `fn new`, builder `fn with_predecessor_correction`), threaded through `Fit::fit` → `fn cluster_optics_xi` → `fn xi_cluster`, which applies `fn correct_predecessor` ONLY when the flag is set — matching sklearn's `predecessor_correction=True` default (`sklearn/cluster/_optics.py:277`) and its conditional use (`if predecessor_correction:` `:1147-1150`). Consumer (R-DEFER-1, non-test): the `Xi` arm of `Fit::fit` reads `self.predecessor_correction`. Guard (live-oracle): `green_xi_predecessor_correction_toggle_small10` — `predecessor_correction=False` changes the corrected end (`[4,7]→[4,8]`) and labels point 2, bit-exact vs sklearn. (The remaining REQ-10 param surface stays NOT-STARTED.) |
//! | REQ-10 (param surface `metric`/`p`/`algorithm`/`leaf_size`/`n_jobs` + `min_samples=5` default + float fractions) | NOT-STARTED | open prereq blocker #1088. sklearn `__init__` (`:266-297`); ferrolearn `fn new(min_samples)` required, only `max_eps`/`xi`/`min_cluster_size` builders. |
//! | REQ-11 (validation accept/reject BOUNDARIES) | SHIPPED | impl: `Fit::fit`'s parameter-validation block now matches sklearn's `OPTICS._parameter_constraints` (`sklearn/cluster/_optics.py:242-264`) at the accept/reject BOUNDARY: `min_samples < 2` REJECTED (`Interval(Integral, 2, None, closed="left")` `:243-246` — `{0,1}` rejected; was permitting `min_samples=1`), `max_eps < 0` REJECTED but `max_eps == 0` ACCEPTED (`Interval(Real, 0, None, closed="both")` `:247` — was over-rejecting `max_eps==0`), `xi < 0 \|\| xi > 1` REJECTED but `xi == 0` AND `xi == 1` ACCEPTED (`Interval(Real, 0, 1, closed="both")` `:253` — was over-rejecting `xi∈{0,1}`). The `max_eps == 0` degenerate path needed NO algorithm change: with `max_eps==0` no point has a neighbour, so `fn core_distance` returns `∞` for all (the `if core_distances[point].is_finite()` guard in `Fit::fit` skips `fn update_seeds`), `reachability_` stays `∞`, `ordering_` is by index, and `fn xi_cluster_extraction` on the all-`∞` plot yields all-`-1` labels (the `ratio = inf/inf = NaN` comparisons are all false → no steep points → no clusters) — value-matching the live oracle WITHOUT divide-by-zero / OOB / unwrap-None (R-CODE-2). `xi == 1` likewise does not panic: ferrolearn's Xi ratios use Rust float arithmetic (`1.0/0.0 == inf`), unlike sklearn's Python `ZeroDivisionError`. **Error TYPE ABI:** the grandfathered crate `FerroError::InvalidParameter` is kept (NOT sklearn's `InvalidParameterError`); only the accept/reject BOUNDARY is matched (R-DEV-2 user-API ABI is the boundary, not the exception class name — a crate-wide grandfathered convention). Consumer (R-DEFER-1, non-test): the validated `Fit::fit` is reachable via the crate re-export `pub use optics::{FittedOPTICS, OPTICS}` (`lib.rs`). Guards (live-oracle, R-CHAR-3): `green_min_samples_below_2_rejected`, `green_min_samples_2_accepted`, `green_max_eps_zero_degenerate_matches_oracle` (element-wise `core_distances_`/`reachability_`/`ordering_`/`labels_` vs the oracle), `green_max_eps_half_unchanged`, `green_max_eps_negative_rejected`, `green_xi_zero_accepted`, `green_xi_one_accepted_no_panic`, `green_xi_negative_rejected`, `green_xi_above_one_rejected` in `tests/divergence_optics.rs`. The `min_samples` float-fraction branch (`Interval(RealNotInt, 0, 1, closed="both")` `:245`) + the remaining param surface (`metric`/`p`/`algorithm`/`leaf_size`/`n_jobs`, the `min_samples=5` default) stay NOT-STARTED (REQ-10, blocker #1088). |
//! | REQ-12 (PyO3 binding) | NOT-STARTED | open prereq blocker #1090. No `_RsOPTICS` (grep empty); `import ferrolearn` cannot reach OPTICS. |
//! | REQ-13 (ferray substrate) | NOT-STARTED | open prereq blocker #1091. `optics.rs` imports `ndarray`/`num-traits`, not `ferray-core` (R-SUBSTRATE-1/2). |

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::Fit;
use ndarray::{Array1, Array2};
use num_traits::Float;
use std::cmp::Ordering;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration struct
// ─────────────────────────────────────────────────────────────────────────────

/// Cluster-extraction method used by [`OPTICS`] to derive `labels_` from the
/// reachability plot.
///
/// Mirrors scikit-learn's `cluster_method` parameter, whose
/// `_parameter_constraints` is `StrOptions({"dbscan", "xi"})` with default
/// `"xi"` (`sklearn/cluster/_optics.py:251`, `:274`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OpticsClusterMethod {
    /// The automatic Xi-steep extraction technique (the default), proposed in
    /// the OPTICS paper. Uses the `xi`/`min_cluster_size` parameters.
    #[default]
    Xi,
    /// A DBSCAN-like extraction at a fixed `eps`, via
    /// [`cluster_optics_dbscan`]. Uses the `eps` parameter.
    Dbscan,
}

/// OPTICS clustering configuration (unfitted).
///
/// Holds hyperparameters.  Call [`Fit::fit`] to run the algorithm and produce
/// a [`FittedOPTICS`].
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct OPTICS<F> {
    /// Minimum number of points required to form a core point (including
    /// the point itself).
    pub min_samples: usize,
    /// Maximum radius considered for neighbourhood queries.  Points beyond
    /// this distance are not considered neighbours.  Defaults to `F::infinity()`.
    pub max_eps: F,
    /// Xi steep-point threshold used by [`FittedOPTICS::extract_clusters`].
    /// A value in `(0, 1)`.  Defaults to `0.05`.
    pub xi: F,
    /// Minimum number of points required for a cluster to be kept.
    /// Clusters smaller than this are relabelled as noise (`-1`).
    /// Defaults to `None`, meaning use `min_samples`.
    pub min_cluster_size: Option<usize>,
    /// The cluster-extraction method used to derive `labels_` from the
    /// reachability plot.  Defaults to [`OpticsClusterMethod::Xi`], matching
    /// scikit-learn's `cluster_method="xi"` (`sklearn/cluster/_optics.py:274`).
    pub cluster_method: OpticsClusterMethod,
    /// DBSCAN `eps` parameter, used only when
    /// `cluster_method == OpticsClusterMethod::Dbscan`.  When `None` (the
    /// default) it assumes the same value as `max_eps`, matching scikit-learn's
    /// `eps=None` (`sklearn/cluster/_optics.py:275`, resolved `:375-378`).
    pub eps: Option<F>,
    /// Correct clusters based on the calculated predecessors during Xi
    /// extraction.  Defaults to `true`, matching scikit-learn's
    /// `predecessor_correction=True` (`sklearn/cluster/_optics.py:277`).
    pub predecessor_correction: bool,
}

impl<F: Float> OPTICS<F> {
    /// Create a new `OPTICS` with the given `min_samples`.
    ///
    /// Defaults: `max_eps = F::infinity()`, `xi = 0.05`, `min_cluster_size = None`.
    #[must_use]
    pub fn new(min_samples: usize) -> Self {
        Self {
            min_samples,
            max_eps: F::infinity(),
            xi: F::from(0.05).unwrap_or_else(|| F::from(5e-2).unwrap()),
            min_cluster_size: None,
            cluster_method: OpticsClusterMethod::Xi,
            eps: None,
            predecessor_correction: true,
        }
    }

    /// Set the maximum neighbourhood radius.
    #[must_use]
    pub fn with_max_eps(mut self, max_eps: F) -> Self {
        self.max_eps = max_eps;
        self
    }

    /// Set the Xi steep-point threshold.
    ///
    /// Must be in `(0, 1)`.
    #[must_use]
    pub fn with_xi(mut self, xi: F) -> Self {
        self.xi = xi;
        self
    }

    /// Set the minimum cluster size.
    ///
    /// Clusters with fewer than `min_cluster_size` points are relabelled as
    /// noise (`-1`).  When `None` (the default), `min_samples` is used as
    /// the minimum cluster size, matching scikit-learn's default behaviour.
    #[must_use]
    pub fn with_min_cluster_size(mut self, size: usize) -> Self {
        self.min_cluster_size = Some(size);
        self
    }

    /// Set the cluster-extraction method (`Xi` or `Dbscan`).
    ///
    /// Mirrors scikit-learn's `cluster_method` parameter
    /// (`sklearn/cluster/_optics.py:274`). With [`OpticsClusterMethod::Dbscan`]
    /// the labels are derived from the reachability plot at a fixed `eps` via
    /// [`cluster_optics_dbscan`]; with [`OpticsClusterMethod::Xi`] (the default)
    /// the automatic Xi-steep extraction is used.
    #[must_use]
    pub fn with_cluster_method(mut self, method: OpticsClusterMethod) -> Self {
        self.cluster_method = method;
        self
    }

    /// Set the DBSCAN `eps` parameter (used only when
    /// `cluster_method == OpticsClusterMethod::Dbscan`).
    ///
    /// When unset (`None`, the default), `eps` assumes the value of `max_eps`,
    /// matching scikit-learn's `eps=None` resolution
    /// (`sklearn/cluster/_optics.py:375-378`). `eps` must not exceed `max_eps`,
    /// otherwise [`Fit::fit`] returns an error.
    #[must_use]
    pub fn with_eps(mut self, eps: F) -> Self {
        self.eps = Some(eps);
        self
    }

    /// Set whether to apply predecessor correction during Xi extraction.
    ///
    /// Mirrors scikit-learn's `predecessor_correction` parameter, a `bool`
    /// with default `True` (`sklearn/cluster/_optics.py:277`). When `true`
    /// (the default), each candidate Xi cluster is adjusted via
    /// `_correct_predecessor` (Algorithm 2 of Schubert & Gertz 2018) before
    /// being accepted.
    #[must_use]
    pub fn with_predecessor_correction(mut self, predecessor_correction: bool) -> Self {
        self.predecessor_correction = predecessor_correction;
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fitted struct
// ─────────────────────────────────────────────────────────────────────────────

/// Fitted OPTICS model.
///
/// Stores the reachability ordering, reachability distances, core distances,
/// predecessor tracking, and cluster labels (extracted via the Xi method).
///
/// OPTICS does **not** implement [`Predict`](ferrolearn_core::Predict).
#[derive(Debug, Clone)]
pub struct FittedOPTICS<F> {
    /// Indices of data points in the reachability order.
    ordering_: Vec<usize>,
    /// Reachability distance for each data point (indexed by original point
    /// index, not by ordering position).  The first point processed always
    /// has reachability distance `∞`.
    reachability_: Array1<F>,
    /// Core distance for each point (indexed by original point index).
    /// Equals `∞` for non-core points.
    core_distances_: Array1<F>,
    /// Cluster label for each training sample (0-indexed for clusters; `-1`
    /// for noise).  Extracted using the Xi method.
    labels_: Array1<isize>,
    /// Predecessor for each point in the OPTICS ordering.
    /// `predecessors_[i] = Some(j)` means point `i` was reached from point `j`.
    /// The first point in each connected component has `None`.
    predecessors_: Vec<Option<usize>>,
    /// sklearn-faithful `predecessor_`: a `-1`-sentinel integer array, indexed by
    /// original sample index. `predecessor_[i] = j` (`>= 0`) means point `i` was
    /// reached from point `j`; seed points have `-1`. Built from `predecessors_`
    /// at fit time (`Some(j) -> j as i64`, `None -> -1`), mirroring
    /// `predecessor_ = np.full(n_samples, -1, dtype=int)` then
    /// `predecessor_[unproc[improved]] = point_index` in `compute_optics_graph`
    /// (`sklearn/cluster/_optics.py:604-605`, `:714`).
    predecessor_: Array1<i64>,
    /// The `min_samples` value used during fitting (needed for Xi extraction).
    min_samples_: usize,
    /// The hierarchy of clusters discovered by the Xi method, as a
    /// `(n_clusters, 2)` integer array of `[start, end]` intervals (both
    /// indices inclusive, in reachability-plot / `ordering_` order).
    ///
    /// Mirrors scikit-learn's `OPTICS.cluster_hierarchy_` — "The list of
    /// clusters in the form of `[start, end]` in each row, with all indices
    /// inclusive. The clusters are ordered according to `(end, -start)`
    /// (ascending) so that larger clusters encompassing smaller clusters come
    /// after such nested smaller clusters" (`sklearn/cluster/_optics.py:191-200`,
    /// returned by `cluster_optics_xi` `:859-865` / `_xi_cluster` `:1166-1172`).
    /// Empty (shape `(0, 2)`) when `cluster_method == Dbscan` (sklearn only
    /// sets `cluster_hierarchy_` on the Xi branch, `:373`).
    cluster_hierarchy_: Array2<i64>,
}

impl<F: Float> FittedOPTICS<F> {
    /// Return the reachability ordering (indices into the training data).
    #[must_use]
    pub fn ordering(&self) -> &[usize] {
        &self.ordering_
    }

    /// Return the reachability distances, indexed by original point index.
    #[must_use]
    pub fn reachability(&self) -> &Array1<F> {
        &self.reachability_
    }

    /// Return the core distances, indexed by original point index.
    #[must_use]
    pub fn core_distances(&self) -> &Array1<F> {
        &self.core_distances_
    }

    /// Return the cluster labels (Xi-method extraction).
    ///
    /// Noise points have label `-1`.
    #[must_use]
    pub fn labels(&self) -> &Array1<isize> {
        &self.labels_
    }

    /// Return the predecessors for each point.
    ///
    /// `predecessors()[i] = Some(j)` means point `i` was reached from point `j`
    /// during the OPTICS ordering phase. The first point in each connected
    /// component has `None`.
    #[must_use]
    pub fn predecessors(&self) -> &[Option<usize>] {
        &self.predecessors_
    }

    /// Return the sklearn-faithful `predecessor_`: a `-1`-sentinel integer array
    /// of shape `(n_samples,)`, indexed by original sample index.
    ///
    /// `predecessor()[i] = j` (`>= 0`) means point `i` was reached from point
    /// `j` during the OPTICS ordering phase; seed points (the first point in
    /// each connected component) have `predecessor()[i] = -1`.
    ///
    /// This mirrors scikit-learn's `OPTICS.predecessor_` — "Point that a sample
    /// was reached from, indexed by object order. Seed points have a predecessor
    /// of -1." (`sklearn/cluster/_optics.py:187-189`, built as
    /// `np.full(n_samples, -1, dtype=int)` in `compute_optics_graph` `:604-605`,
    /// `:714`). It is the `-1`-sentinel `Array1<i64>` counterpart of the internal
    /// [`predecessors`](Self::predecessors) (`&[Option<usize>]`, seed `None`).
    #[must_use]
    pub fn predecessor(&self) -> &Array1<i64> {
        &self.predecessor_
    }

    /// Return the cluster hierarchy discovered by the Xi method.
    ///
    /// A `(n_clusters, 2)` integer array of `[start, end]` intervals (both
    /// indices inclusive, in reachability-plot / `ordering_` order), ordered so
    /// that larger clusters encompassing smaller ones come after the nested
    /// smaller clusters. Because `labels_` keeps only the leaf-level clusters,
    /// `cluster_hierarchy().nrows()` is usually greater than the number of
    /// distinct non-noise labels.
    ///
    /// Mirrors scikit-learn's `OPTICS.cluster_hierarchy_`
    /// (`sklearn/cluster/_optics.py:191-200`). Empty (shape `(0, 2)`) when the
    /// `Dbscan` cluster method was used (sklearn only sets it on the Xi branch).
    #[must_use]
    pub fn cluster_hierarchy(&self) -> &Array2<i64> {
        &self.cluster_hierarchy_
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

    /// Extract flat clusters from the reachability plot using the Xi method.
    ///
    /// The Xi method identifies *steep up* and *steep down* areas in the
    /// reachability plot.  Clusters are formed between matching steep-down /
    /// steep-up pairs.
    ///
    /// `xi` must be in `(0, 1)`.  Returns a vector of cluster labels
    /// (length == `n_samples`); noise has label `-1`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if `xi` is outside `(0, 1)`.
    pub fn extract_clusters(&self, xi: F) -> Result<Array1<isize>, FerroError> {
        if xi <= F::zero() || xi >= F::one() {
            return Err(FerroError::InvalidParameter {
                name: "xi".into(),
                reason: "must be in (0, 1)".into(),
            });
        }
        // Re-run the Xi extraction with this `xi`, the fitted `min_samples`
        // (also used as the default `min_cluster_size`), and predecessor
        // correction enabled (the sklearn default). Discards the hierarchy.
        let (labels, _hierarchy) = cluster_optics_xi(
            &self.reachability_,
            &self.predecessor_,
            &self.ordering_,
            xi,
            self.min_samples_,
            None,
            true,
        );
        Ok(labels)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Euclidean distance between two slices.
#[inline]
fn euclidean<F: Float>(a: &[F], b: &[F]) -> F {
    a.iter()
        .zip(b)
        .fold(F::zero(), |acc, (&ai, &bi)| acc + (ai - bi) * (ai - bi))
        .sqrt()
}

/// Return all neighbours of `idx` within distance `max_eps` (sorted by distance).
///
/// Returns `(neighbor_indices, distances)` in ascending distance order.
fn get_neighbors<F: Float>(x: &Array2<F>, idx: usize, max_eps: F) -> (Vec<usize>, Vec<F>) {
    let row = x.row(idx);
    let rs = row.as_slice().unwrap_or(&[]);
    let mut pairs: Vec<(F, usize)> = (0..x.nrows())
        .filter_map(|j| {
            let other = x.row(j);
            let os = other.as_slice().unwrap_or(&[]);
            let d = euclidean(rs, os);
            if d <= max_eps && j != idx {
                Some((d, j))
            } else {
                None
            }
        })
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    let indices = pairs.iter().map(|p| p.1).collect();
    let dists = pairs.iter().map(|p| p.0).collect();
    (indices, dists)
}

/// Compute core distance of `idx`: distance to the `min_samples`-th nearest
/// neighbour within `max_eps`.  Returns `F::infinity()` if fewer than
/// `min_samples` neighbours exist.
fn core_distance<F: Float>(x: &Array2<F>, idx: usize, max_eps: F, min_samples: usize) -> F {
    let row = x.row(idx);
    let rs = row.as_slice().unwrap_or(&[]);

    let mut dists: Vec<F> = (0..x.nrows())
        .filter_map(|j| {
            if j == idx {
                return None;
            }
            let other = x.row(j);
            let os = other.as_slice().unwrap_or(&[]);
            let d = euclidean(rs, os);
            if d <= max_eps { Some(d) } else { None }
        })
        .collect();

    if dists.len() < min_samples.saturating_sub(1) {
        // Not enough neighbours (need min_samples - 1 others).
        return F::infinity();
    }

    dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    // The core distance is the distance to the (min_samples-1)-th other point
    // (0-indexed), i.e., the min_samples-th point overall including self.
    let k = min_samples.saturating_sub(1);
    if k == 0 {
        F::zero()
    } else if k <= dists.len() {
        dists[k - 1]
    } else {
        F::infinity()
    }
}

/// Round a reachability value to the floating-point type's decimal precision,
/// matching sklearn's `np.around(rdists, decimals=np.finfo(dtype).precision)`
/// (`sklearn/cluster/_optics.py:711`).
///
/// `np.finfo` precision is 15 decimals for float64 and 6 for float32.  Rounding
/// collapses sub-ULP differences between candidate reachabilities into exact
/// ties, which is required for sklearn's smallest-index tie-break in the
/// traversal to reproduce its ordering.
#[inline]
fn round_to_precision<F: Float>(v: F) -> F {
    if !v.is_finite() {
        return v;
    }
    let decimals = if std::mem::size_of::<F>() == 4 { 6 } else { 15 };
    let factor = F::from(10f64.powi(decimals)).unwrap_or_else(F::one);
    // numpy's `np.around` uses round-half-to-even (`np.rint`); matching it is
    // load-bearing — half-away-from-zero would shift `538516480713450.5` to
    // `...451` instead of `...450`, breaking the tie collapse. f64/f32 both
    // round through f64 here, then convert back, mirroring numpy's multiply/
    // rint/divide.
    let scaled = (v * factor).to_f64().unwrap_or(0.0).round_ties_even();
    F::from(scaled).unwrap_or(v) / factor
}

/// Update reachability distances and predecessors for the unprocessed
/// neighbours of `current_point`.
///
/// For each unprocessed neighbour `q`, the new reachability distance is
/// `round(max(core_dist_p, dist(p, q)))` (rounded to the dtype precision,
/// matching `np.around` in `_set_reach_dist`, `sklearn/cluster/_optics.py:710-714`).
/// If this improves the current value, `reachability[q]` is lowered and
/// `current_point` is recorded as the predecessor of `q`.
fn update_seeds<F: Float>(
    core_dist_p: F,
    current_point: usize,
    neighbors: &[usize],
    neighbor_dists: &[F],
    processed: &[bool],
    reachability: &mut Array1<F>,
    predecessors: &mut [Option<usize>],
) {
    for (i, &q) in neighbors.iter().enumerate() {
        if processed[q] {
            continue;
        }
        let new_reach = if core_dist_p > neighbor_dists[i] {
            core_dist_p
        } else {
            neighbor_dists[i]
        };
        // np.around to the dtype precision (`_optics.py:711`) so that
        // sub-ULP differences become exact ties for the tie-break.
        let new_reach = round_to_precision(new_reach);
        if new_reach < reachability[q] {
            reachability[q] = new_reach;
            predecessors[q] = Some(current_point);
        }
    }
}

#[derive(Debug, Clone)]
struct OpticsGraph<F> {
    ordering: Vec<usize>,
    core_distances: Array1<F>,
    reachability: Array1<F>,
    predecessors: Vec<Option<usize>>,
    predecessor: Array1<i64>,
}

/// Compute the OPTICS reachability graph.
///
/// This is the ferrolearn translation of scikit-learn's
/// `compute_optics_graph`. It returns the same four graph outputs, indexed by
/// original sample order except for `ordering`: `(ordering, core_distances,
/// reachability, predecessor)`.
///
/// The current Rust surface covers ferrolearn's Euclidean OPTICS subset:
/// `x` is a dense feature matrix, `min_samples` is an integer count, and
/// `max_eps` is the maximum Euclidean neighbor radius.
///
/// # Errors
///
/// Returns [`FerroError::InvalidParameter`] when `min_samples < 2`,
/// `min_samples > n_samples`, or `max_eps` is negative or NaN. Returns
/// [`FerroError::InsufficientSamples`] for an empty input matrix.
pub fn compute_optics_graph<F: Float>(
    x: &Array2<F>,
    min_samples: usize,
    max_eps: F,
) -> Result<(Vec<usize>, Array1<F>, Array1<F>, Array1<i64>), FerroError> {
    let graph = compute_optics_graph_internal(x, min_samples, max_eps)?;
    Ok((
        graph.ordering,
        graph.core_distances,
        graph.reachability,
        graph.predecessor,
    ))
}

fn compute_optics_graph_internal<F: Float>(
    x: &Array2<F>,
    min_samples: usize,
    max_eps: F,
) -> Result<OpticsGraph<F>, FerroError> {
    let n_samples = x.nrows();
    if min_samples < 2 {
        return Err(FerroError::InvalidParameter {
            name: "min_samples".into(),
            reason: "must be at least 2".into(),
        });
    }
    if max_eps < F::zero() || max_eps.is_nan() {
        return Err(FerroError::InvalidParameter {
            name: "max_eps".into(),
            reason: "must be non-negative".into(),
        });
    }
    if n_samples == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: "OPTICS requires at least 1 sample".into(),
        });
    }
    if min_samples > n_samples {
        return Err(FerroError::InvalidParameter {
            name: "min_samples".into(),
            reason: format!(
                "must be no greater than the number of samples ({n_samples}). Got {min_samples}"
            ),
        });
    }

    let mut reachability = Array1::from_elem(n_samples, F::infinity());
    let mut core_distances = Array1::from_elem(n_samples, F::infinity());
    let mut processed = vec![false; n_samples];
    let mut ordering: Vec<usize> = Vec::with_capacity(n_samples);
    let mut predecessors: Vec<Option<usize>> = vec![None; n_samples];

    for i in 0..n_samples {
        core_distances[i] = core_distance(x, i, max_eps, min_samples);
    }

    for _ in 0..n_samples {
        let mut point: Option<usize> = None;
        let mut best = F::infinity();
        for j in 0..n_samples {
            if processed[j] {
                continue;
            }
            if point.is_none() || reachability[j] < best {
                best = reachability[j];
                point = Some(j);
            }
        }
        let Some(point) = point else {
            break;
        };

        processed[point] = true;
        ordering.push(point);

        if core_distances[point].is_finite() {
            let (nbrs, nbr_dists) = get_neighbors(x, point, max_eps);
            update_seeds(
                core_distances[point],
                point,
                &nbrs,
                &nbr_dists,
                &processed,
                &mut reachability,
                &mut predecessors,
            );
        }
    }

    let predecessor: Array1<i64> = predecessors
        .iter()
        .map(|p| p.map_or(-1i64, |j| j as i64))
        .collect();

    Ok(OpticsGraph {
        ordering,
        core_distances,
        reachability,
        predecessors,
        predecessor,
    })
}

/// Perform DBSCAN extraction from an OPTICS reachability graph at a fixed `eps`.
///
/// This is the ferrolearn translation of scikit-learn's
/// `cluster_optics_dbscan(*, reachability, core_distances, ordering, eps)`
/// (`sklearn/cluster/_optics.py:726-788`). Given the OPTICS-computed
/// `reachability_`, `core_distances_`, and `ordering_`, it labels each sample as
/// a member of a DBSCAN-like cluster (`0, 1, 2, …`) or as noise (`-1`), running
/// in linear time over `ordering`.
///
/// The exact two-step labelling (`_optics.py:781-787`):
/// 1. `far_reach = reachability > eps` (STRICT `>`), `near_core =
///    core_distances <= eps` (INCLUSIVE `<=`). Walking the points in `ordering`
///    order, maintain a running cumulative sum of `far_reach[o] & near_core[o]`
///    and assign `labels[o] = cumsum_so_far - 1`. This assigns ALL points.
/// 2. For every point `i` with `far_reach[i] & !near_core[i]`, OVERWRITE
///    `labels[i] = -1` (the noise mask).
///
/// `reachability`, `core_distances` are indexed by ORIGINAL sample index;
/// `ordering` lists the original indices in OPTICS reachability order. The
/// returned labels are indexed by original sample index, shape `(n_samples,)`.
///
/// # Panics
///
/// Does not panic. Out-of-range entries in `ordering` (which OPTICS never
/// produces) are skipped rather than indexing out of bounds.
#[must_use]
pub fn cluster_optics_dbscan<F: Float>(
    reachability: &[F],
    core_distances: &[F],
    ordering: &[usize],
    eps: F,
) -> Array1<isize> {
    let n_samples = core_distances.len();
    // labels = np.zeros(n_samples, dtype=int)  (`_optics.py:782`).
    let mut labels = Array1::<isize>::zeros(n_samples);

    // Step 1: labels[ordering] = np.cumsum(far_reach[ordering] & near_core[ordering]) - 1
    // (`_optics.py:784-786`). `far_reach` uses STRICT `>`, `near_core` uses
    // INCLUSIVE `<=`. We walk `ordering` accumulating the boolean cumsum.
    let mut cumsum: isize = 0;
    for &o in ordering {
        if o >= n_samples {
            // OPTICS never emits an out-of-range index; guard defensively.
            continue;
        }
        let far_reach = reachability[o] > eps;
        let near_core = core_distances[o] <= eps;
        if far_reach && near_core {
            cumsum += 1;
        }
        labels[o] = cumsum - 1;
    }

    // Step 2: labels[far_reach & ~near_core] = -1  (`_optics.py:787`). This
    // OVERWRITES the cumsum value for the noise points, indexed by original
    // sample index over ALL samples (not just those in `ordering`).
    for i in 0..n_samples {
        let far_reach = reachability[i] > eps;
        let near_core = core_distances[i] <= eps;
        if far_reach && !near_core {
            labels[i] = -1;
        }
    }

    labels
}

// ─────────────────────────────────────────────────────────────────────────────
// Xi-method cluster extraction — faithful port of sklearn's
// `cluster_optics_xi` / `_xi_cluster` (`sklearn/cluster/_optics.py:810-1202`).
//
// The reachability plot, predecessor plot, and cluster intervals all index the
// reachability-PLOT (i.e. `ordering`) order, NOT the original sample order. The
// final `_extract_xi_labels` scatters back to original-sample order.
// ─────────────────────────────────────────────────────────────────────────────

/// A steep-down area (SDA) tracked during Xi extraction
/// (a `{"start", "end", "mib"}` dict in sklearn, `_optics.py:1104`).
#[derive(Debug, Clone)]
struct SteepDownArea {
    start: usize,
    end: usize,
    mib: f64,
}

/// Automatically extract clusters via the Xi-steep method.
///
/// Faithful translation of scikit-learn's `cluster_optics_xi`
/// (`sklearn/cluster/_optics.py:810-918`). Resolves `min_cluster_size`
/// (`None` → `min_samples`), runs [`xi_cluster`] over the reachability plot, and
/// derives leaf labels via [`extract_xi_labels`].
///
/// Returns `(labels, clusters)` where `labels` is the `(n_samples,)` cluster
/// assignment (`-1` for noise, indexed by ORIGINAL sample order) and `clusters`
/// is the `(n_clusters, 2)` `[start, end]` hierarchy (PLOT-order intervals,
/// both inclusive) — exactly `OPTICS.labels_` and `OPTICS.cluster_hierarchy_`.
///
/// `reachability` and `predecessor` are indexed by ORIGINAL sample order;
/// `ordering` lists original sample indices in reachability-plot order.
///
/// ferrolearn's `min_samples` / `min_cluster_size` are already `>= 2` integers
/// (the float-fraction `<= 1` resolution in `cluster_optics_xi` `:900-906` is
/// REQ-10, out of scope), so only the `None → min_samples` default is applied
/// here.
pub fn cluster_optics_xi<F: Float>(
    reachability: &Array1<F>,
    predecessor: &Array1<i64>,
    ordering: &[usize],
    xi: F,
    min_samples: usize,
    min_cluster_size: Option<usize>,
    predecessor_correction: bool,
) -> (Array1<isize>, Array2<i64>) {
    let n_total = reachability.len();
    // `if min_cluster_size is None: min_cluster_size = min_samples` (`:902-903`).
    let min_cluster_size = min_cluster_size.unwrap_or(min_samples);

    if ordering.is_empty() {
        return (
            Array1::from_elem(n_total, -1isize),
            Array2::<i64>::zeros((0, 2)),
        );
    }

    // reachability_plot = reachability[ordering] (`:909`), as f64 with non-finite
    // → +inf. _xi_cluster then hstacks an extra +inf sentinel (`:1070`).
    let mut r_plot: Vec<f64> = ordering
        .iter()
        .map(|&i| {
            let v = reachability[i].to_f64().unwrap_or(f64::INFINITY);
            if v.is_finite() { v } else { f64::INFINITY }
        })
        .collect();
    r_plot.push(f64::INFINITY);

    // predecessor_plot = predecessor[ordering] (`:910`), keeping sklearn's `-1`
    // sentinel for seed points.
    let pred_plot: Vec<i64> = ordering.iter().map(|&i| predecessor[i]).collect();

    let xi_f64 = xi.to_f64().unwrap_or(0.05);

    let clusters = xi_cluster(
        &r_plot,
        &pred_plot,
        ordering,
        xi_f64,
        min_samples,
        min_cluster_size,
        predecessor_correction,
    );

    let labels = extract_xi_labels(ordering, &clusters, n_total);

    // Build the `(n_clusters, 2)` hierarchy array (`np.array(clusters)`, `:1172`).
    let hierarchy = if clusters.is_empty() {
        Array2::<i64>::zeros((0, 2))
    } else {
        let mut flat: Vec<i64> = Vec::with_capacity(clusters.len() * 2);
        for &(s, e) in &clusters {
            flat.push(s as i64);
            flat.push(e as i64);
        }
        Array2::from_shape_vec((clusters.len(), 2), flat)
            .unwrap_or_else(|_| Array2::<i64>::zeros((0, 2)))
    };

    (labels, hierarchy)
}

/// Extend the steep area until it's maximal.
///
/// Faithful translation of `_extend_region` (`sklearn/cluster/_optics.py:921-981`).
/// The same function serves both directions: to extend an upward region pass
/// `steep_point=steep_upward, xward_point=downward`; to extend a downward region
/// pass `steep_point=steep_downward, xward_point=upward`. Up/down steep regions
/// can't have more than `min_samples` consecutive non-steep points going the
/// correct direction. The returned `end` is inclusive.
fn extend_region(
    steep_point: &[bool],
    xward_point: &[bool],
    start: usize,
    min_samples: usize,
) -> usize {
    let n_samples = steep_point.len();
    let mut non_xward_points = 0usize;
    let mut index = start;
    let mut end = start;
    // find a maximal area
    while index < n_samples {
        if steep_point[index] {
            non_xward_points = 0;
            end = index;
        } else if !xward_point[index] {
            // it's not a steep point, but still goes up.
            non_xward_points += 1;
            // region should include no more than min_samples consecutive
            // non steep xward points.
            if non_xward_points > min_samples {
                break;
            }
        } else {
            return end;
        }
        index += 1;
    }
    end
}

/// Update steep-down areas (SDAs) using the new maximum-in-between (`mib`) value
/// and `xi_complement = 1 - xi`.
///
/// Faithful translation of `_update_filter_sdas`
/// (`sklearn/cluster/_optics.py:984-995`): if `mib` is `inf` every SDA is
/// dropped; otherwise keep SDAs whose `r_plot[start] * xi_complement >= mib`
/// (sklearn `mib <= reachability_plot[sda["start"]] * xi_complement`) and update
/// each survivor's `mib = max(mib, sda.mib)`.
fn update_filter_sdas(
    sdas: Vec<SteepDownArea>,
    mib: f64,
    xi_complement: f64,
    r_plot: &[f64],
) -> Vec<SteepDownArea> {
    if mib.is_infinite() {
        return Vec::new();
    }
    let mut res: Vec<SteepDownArea> = sdas
        .into_iter()
        .filter(|sda| mib <= r_plot[sda.start] * xi_complement)
        .collect();
    for sda in &mut res {
        sda.mib = sda.mib.max(mib);
    }
    res
}

/// Correct for predecessors (Algorithm 2 of Schubert & Gertz 2018).
///
/// Faithful translation of `_correct_predecessor`
/// (`sklearn/cluster/_optics.py:998-1017`). Inputs are PLOT-ordered. Returns
/// `Some((s, e))` for the corrected interval, or `None` (sklearn `(None, None)`)
/// when the interval collapses (`s >= e`).
///
/// `predecessor_plot[e]` is an ORIGINAL sample index (or `-1` for a seed);
/// `ordering[i]` is also an original sample index, so the membership test
/// `p_e == ordering[i]` compares like-for-like. A `-1` predecessor never equals
/// any `ordering[i]` (all `>= 0`), matching sklearn where a seed's predecessor
/// (`-1`) is not in the ordering window.
fn correct_predecessor(
    r_plot: &[f64],
    predecessor_plot: &[i64],
    ordering: &[usize],
    s: usize,
    mut e: usize,
) -> Option<(usize, usize)> {
    let mut s = s;
    while s < e {
        if r_plot[s] > r_plot[e] {
            return Some((s, e));
        }
        let p_e = predecessor_plot[e];
        let mut found = false;
        for &item in ordering.iter().take(e).skip(s) {
            if p_e == item as i64 {
                found = true;
                break;
            }
        }
        if found {
            return Some((s, e));
        }
        e -= 1;
        // (s is never advanced by sklearn; kept mutable only to mirror the
        // signature — the `while s < e` guard handles the collapse.)
        let _ = &mut s;
    }
    None
}

/// Automatically extract clusters according to the Xi-steep method (Figure 19).
///
/// Faithful translation of `_xi_cluster` (`sklearn/cluster/_optics.py:1020-1172`).
///
/// `r_plot` already has the appended `+inf` sentinel (length `n_plot + 1`).
/// `predecessor_plot` is PLOT-ordered with the `-1` seed sentinel. Returns the
/// list of `(start, end)` cluster intervals (PLOT-order, both inclusive) in
/// discovery order, with each steep-up area's matches reversed (smaller clusters
/// first), exactly as sklearn returns `np.array(clusters)`.
///
/// The ratio arithmetic mirrors numpy under `errstate(invalid="ignore")`:
/// `inf/inf` and `0/0` give `NaN` (all comparisons false), `x/inf = 0`,
/// finite `x/0 = +inf`. Rust's IEEE-754 float division reproduces this exactly,
/// so no special-casing is needed (and no divide-by-zero panic, R-CODE-2).
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors sklearn `_xi_cluster`'s 7-parameter signature (sklearn/cluster/_optics.py:1020-1028)"
)]
fn xi_cluster(
    r_plot: &[f64],
    predecessor_plot: &[i64],
    ordering: &[usize],
    xi: f64,
    min_samples: usize,
    min_cluster_size: usize,
    predecessor_correction: bool,
) -> Vec<(usize, usize)> {
    // xi_complement = 1 - xi (`:1072`).
    let xi_complement = 1.0 - xi;

    let n_plot = r_plot.len() - 1; // excluding the appended +inf sentinel

    // Definition 9 (paper-corrected): steep DOWN uses `>=`, steep UP uses `<=`.
    // ratio = r_plot[:-1] / r_plot[1:] (`:1083-1087`).
    let mut steep_upward = vec![false; n_plot];
    let mut steep_downward = vec![false; n_plot];
    let mut upward = vec![false; n_plot];
    let mut downward = vec![false; n_plot];
    let inv_xi_complement = 1.0 / xi_complement; // 1 / (1 - xi); +inf when xi == 1
    for i in 0..n_plot {
        let ratio = r_plot[i] / r_plot[i + 1]; // NaN for inf/inf or 0/0
        steep_upward[i] = ratio <= xi_complement;
        steep_downward[i] = ratio >= inv_xi_complement;
        downward[i] = ratio > 1.0;
        upward[i] = ratio < 1.0;
    }

    let mut sdas: Vec<SteepDownArea> = Vec::new();
    let mut clusters: Vec<(usize, usize)> = Vec::new();
    let mut index = 0usize;
    let mut mib = 0.0_f64; // maximum in between

    // for steep_index in np.flatnonzero(steep_upward | steep_downward) (`:1091`).
    let steep_indices: Vec<usize> = (0..n_plot)
        .filter(|&i| steep_upward[i] || steep_downward[i])
        .collect();

    for steep_index in steep_indices {
        // continue if steep_index is inside a discovered xward area (`:1094-1095`).
        if steep_index < index {
            continue;
        }

        // mib = max(mib, np.max(r_plot[index : steep_index + 1])) (`:1097`).
        for &item in r_plot.iter().take(steep_index + 1).skip(index) {
            if item > mib {
                mib = item;
            }
        }

        if steep_downward[steep_index] {
            // --- steep downward area (`:1100-1107`) ---
            sdas = update_filter_sdas(sdas, mib, xi_complement, r_plot);
            let d_start = steep_index;
            let d_end = extend_region(&steep_downward, &upward, d_start, min_samples);
            sdas.push(SteepDownArea {
                start: d_start,
                end: d_end,
                mib: 0.0,
            });
            index = d_end + 1;
            mib = r_plot[index];
        } else {
            // --- steep upward area (`:1110-1170`) ---
            sdas = update_filter_sdas(sdas, mib, xi_complement, r_plot);
            let u_start = steep_index;
            let u_end = extend_region(&steep_upward, &downward, u_start, min_samples);
            index = u_end + 1;
            mib = r_plot[index];

            let mut u_clusters: Vec<(usize, usize)> = Vec::new();
            for d in &sdas {
                let mut c_start = d.start;
                let mut c_end = u_end;

                // line (**), sc2*: r_plot[c_end + 1] * xi_complement < D["mib"]
                // → continue (`:1123-1124`). c_end+1 <= u_end+1 = index <= n_plot,
                // and r_plot has the appended sentinel, so the index is in range.
                if r_plot[c_end + 1] * xi_complement < d.mib {
                    continue;
                }

                // Definition 11: criterion 4 (`:1126-1144`).
                let d_max = r_plot[d.start];
                if d_max * xi_complement >= r_plot[c_end + 1] {
                    // Find the first index from the left almost at the end level.
                    // sklearn evaluates `r_plot[c_start+1] > r_plot[c_end+1]`
                    // FIRST then `c_start < D["end"]` (short-circuit `and`). We
                    // guard `c_start + 1 < r_plot.len()` defensively (always true:
                    // c_start < D["end"] <= n_plot-1 here), no OOB.
                    while c_start < d.end && r_plot[c_start + 1] > r_plot[c_end + 1] {
                        c_start += 1;
                    }
                } else if r_plot[c_end + 1] * xi_complement >= d_max {
                    // Find the first index from the right almost at the start
                    // level. Paper-correction: `r(x) > r(sD)` (not `<`). sklearn:
                    // `while r_plot[c_end - 1] > D_max and c_end > U_start`.
                    while c_end > u_start && r_plot[c_end - 1] > d_max {
                        c_end -= 1;
                    }
                }

                // predecessor correction (`:1146-1152`).
                if predecessor_correction {
                    match correct_predecessor(r_plot, predecessor_plot, ordering, c_start, c_end) {
                        Some((cs, ce)) => {
                            c_start = cs;
                            c_end = ce;
                        }
                        None => continue, // c_start is None → continue
                    }
                }

                // Definition 11: criterion 3.a (REQ-7 — INSIDE the loop):
                // c_end - c_start + 1 < min_cluster_size → continue (`:1154-1156`).
                if c_end + 1 - c_start < min_cluster_size {
                    continue;
                }

                // Definition 11: criterion 1: c_start > D["end"] → continue (`:1158-1160`).
                if c_start > d.end {
                    continue;
                }

                // Definition 11: criterion 2: c_end < U_start → continue (`:1162-1164`).
                if c_end < u_start {
                    continue;
                }

                u_clusters.push((c_start, c_end));
            }

            // add smaller clusters first (`:1168-1170`).
            u_clusters.reverse();
            clusters.extend(u_clusters);
        }
    }

    clusters
}

/// Extract labels from the clusters returned by [`xi_cluster`].
///
/// Faithful translation of `_extract_xi_labels`
/// (`sklearn/cluster/_optics.py:1175-1201`). Relies on `clusters` being stored
/// smaller-first: a cluster interval `[c0, c1]` (PLOT-order) is given the next
/// label only if NONE of its plot positions is already labelled, selecting the
/// leaf-level clusters. Then `labels[ordering] = labels.copy()` scatters from
/// plot order back to ORIGINAL sample order.
fn extract_xi_labels(
    ordering: &[usize],
    clusters: &[(usize, usize)],
    n_total: usize,
) -> Array1<isize> {
    let n = ordering.len();
    // labels = np.full(len(ordering), -1) (`:1194`). Length is n_ordered; the
    // plot positions index [0, n). The scatter restores original-sample shape.
    let mut plot_labels = vec![-1isize; n];
    let mut label = 0isize;
    for &(c0, c1) in clusters {
        // `if not np.any(labels[c0 : c1+1] != -1)` (`:1197`). Guard the slice
        // against any out-of-range interval (xi_cluster never emits one).
        let end = (c1 + 1).min(n);
        let start = c0.min(end);
        let any_assigned = plot_labels[start..end].iter().any(|&v| v != -1);
        if !any_assigned {
            for v in &mut plot_labels[start..end] {
                *v = label;
            }
            label += 1;
        }
    }

    // labels[ordering] = labels.copy() (`:1200`): scatter plot-order labels back
    // to original-sample order. `labels` (original order) starts all -1; any
    // sample not in `ordering` (OPTICS always covers all) stays -1.
    let mut labels = Array1::from_elem(n_total, -1isize);
    for (plot_pos, &sample) in ordering.iter().enumerate() {
        labels[sample] = plot_labels[plot_pos];
    }
    labels
}

// ─────────────────────────────────────────────────────────────────────────────
// Fit implementation
// ─────────────────────────────────────────────────────────────────────────────

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for OPTICS<F> {
    type Fitted = FittedOPTICS<F>;
    type Error = FerroError;

    /// Fit the OPTICS model to the data.
    ///
    /// Computes the reachability ordering and distances for all training points.
    /// Cluster labels are extracted using the Xi method with the configured `xi`
    /// parameter.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `min_samples < 2`, `max_eps < 0`,
    ///   or `xi` is outside `[0, 1]`.
    /// - [`FerroError::InsufficientSamples`] if the dataset is empty.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedOPTICS<F>, FerroError> {
        let n_samples = x.nrows();

        // Validate parameters — boundaries match sklearn's
        // `OPTICS._parameter_constraints` (`sklearn/cluster/_optics.py:242-264`).
        // The error TYPE stays the grandfathered crate `FerroError` ABI (not
        // sklearn's `InvalidParameterError`); only the accept/reject BOUNDARY is
        // matched.
        //
        // `"min_samples": [Interval(Integral, 2, None, closed="left"), ...]`
        // (`:243-246`) — an int `min_samples` must be `>= 2`, so `{0, 1}` are
        // rejected. (ferrolearn's `min_samples` is `usize`; the float-fraction
        // branch is REQ-10, out of scope.)
        if self.min_samples < 2 {
            return Err(FerroError::InvalidParameter {
                name: "min_samples".into(),
                reason: "must be at least 2".into(),
            });
        }
        // `"max_eps": [Interval(Real, 0, None, closed="both")]` (`:247`) —
        // `max_eps >= 0`, with `max_eps == 0` ALLOWED (closed at 0). Reject only
        // `max_eps < 0` (for an `F` that can be negative). With `max_eps == 0`
        // the fit RUNS: no point has a neighbour within `max_eps`, so every core
        // distance and reachability is `∞`, ordering is by index, and the labels
        // are all `-1` (matching the live sklearn 1.5.2 oracle).
        // `NaN` is rejected (NaN ∉ [0, ∞]; sklearn's `Interval` rejects it as
        // `InvalidParameterError`). Note `NaN < 0` is `false`, so the NaN guard
        // is explicit (#2197).
        if self.max_eps < F::zero() || self.max_eps.is_nan() {
            return Err(FerroError::InvalidParameter {
                name: "max_eps".into(),
                reason: "must be non-negative".into(),
            });
        }
        // `"xi": [Interval(Real, 0, 1, closed="both")]` (`:253`) — `xi ∈ [0, 1]`,
        // BOTH endpoints allowed (`xi == 0` and `xi == 1` accepted by parameter
        // validation). Reject only `xi < 0 || xi > 1`. (sklearn's `_xi_cluster`
        // raises a runtime `ZeroDivisionError` for `xi == 1.0` because
        // `1 - xi == 0`; ferrolearn's Xi extraction computes the same ratios in
        // Rust float arithmetic — `1.0/0.0 == inf`, `inf/inf == NaN` — so it does
        // NOT panic (R-CODE-2), producing all-`-1` labels. Matching the Xi
        // `labels_` VALUE for these endpoints is REQ-5, NOT-STARTED; only the
        // accept/reject BOUNDARY is matched here.)
        if self.xi < F::zero() || self.xi > F::one() || self.xi.is_nan() {
            return Err(FerroError::InvalidParameter {
                name: "xi".into(),
                reason: "must be in [0, 1]".into(),
            });
        }
        // `xi == 1.0` passes sklearn's `_parameter_constraints` (closed at 1) but
        // makes `_xi_cluster` divide by `1 - xi == 0`, raising a runtime
        // `ZeroDivisionError` IN THE XI METHOD (`_optics.py` steep-area ratio);
        // the OBSERVABLE contract is therefore "fit raises". We mirror that by
        // erroring for `xi == 1` ONLY on the Xi path (the `'dbscan'` method
        // ignores `xi`, so sklearn does NOT raise there). #2197.
        if matches!(self.cluster_method, OpticsClusterMethod::Xi) && self.xi == F::one() {
            return Err(FerroError::InvalidParameter {
                name: "xi".into(),
                reason: "xi == 1 is not supported by the Xi method (1 - xi == 0)".into(),
            });
        }
        // `"eps": [Interval(Real, 0, None, closed="both"), None]` (`:252`) — when
        // set, `eps >= 0` (NaN rejected). sklearn validates this BEFORE the fit
        // body, so it applies regardless of `cluster_method` (the Xi path never
        // reads `eps`, but an invalid `eps` is still rejected). #2198.
        if self.eps.is_some_and(|eps| eps < F::zero() || eps.is_nan()) {
            return Err(FerroError::InvalidParameter {
                name: "eps".into(),
                reason: "must be non-negative".into(),
            });
        }

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "OPTICS requires at least 1 sample".into(),
            });
        }
        // sklearn `_validate_size(min_samples, n_samples, "min_samples")`
        // (`_optics.py:393-400`, called `:597`): `min_samples` must be NO GREATER
        // than `n_samples` (else `ValueError`). ferrolearn previously fit silently
        // with all-`∞` core distances (#2197).
        if self.min_samples > n_samples {
            return Err(FerroError::InvalidParameter {
                name: "min_samples".into(),
                reason: format!(
                    "must be no greater than the number of samples ({n_samples}). Got {}",
                    self.min_samples
                ),
            });
        }

        let graph = compute_optics_graph_internal(x, self.min_samples, self.max_eps)?;

        // Extract cluster labels.  sklearn dispatches on `cluster_method`
        // (`sklearn/cluster/_optics.py:363-390`): `"xi"` runs `cluster_optics_xi`
        // (the default), `"dbscan"` resolves `eps` then runs
        // `cluster_optics_dbscan`. The Xi branch additionally sets
        // `cluster_hierarchy_` (`:373`); the dbscan branch leaves it empty.
        let (labels, cluster_hierarchy_) = match self.cluster_method {
            OpticsClusterMethod::Xi => {
                // labels_, clusters_ = cluster_optics_xi(...) (`:364-372`);
                // self.cluster_hierarchy_ = clusters_ (`:373`).
                cluster_optics_xi(
                    &graph.reachability,
                    &graph.predecessor,
                    &graph.ordering,
                    self.xi,
                    self.min_samples,
                    self.min_cluster_size,
                    self.predecessor_correction,
                )
            }
            OpticsClusterMethod::Dbscan => {
                // Resolve `eps`: `eps = self.eps if self.eps is not None else
                // self.max_eps` (`sklearn/cluster/_optics.py:375-378`).
                let eps = self.eps.unwrap_or(self.max_eps);
                // `if eps > self.max_eps: raise ValueError("Specify an epsilon
                // smaller than %s. Got %s.")` (`:380-383`).
                if eps > self.max_eps {
                    return Err(FerroError::InvalidParameter {
                        name: "eps".into(),
                        reason: format!(
                            "Specify an epsilon smaller than {:?}. Got {:?}.",
                            self.max_eps.to_f64().unwrap_or(f64::INFINITY),
                            eps.to_f64().unwrap_or(f64::INFINITY)
                        ),
                    });
                }
                // labels_ = cluster_optics_dbscan(reachability_, core_distances_,
                // ordering_, eps)  (`:385-390`). sklearn does NOT set
                // `cluster_hierarchy_` on this branch, so it stays empty.
                let labels = cluster_optics_dbscan(
                    graph.reachability.as_slice().unwrap_or(&[]),
                    graph.core_distances.as_slice().unwrap_or(&[]),
                    &graph.ordering,
                    eps,
                );
                (labels, Array2::<i64>::zeros((0, 2)))
            }
        };

        Ok(FittedOPTICS {
            ordering_: graph.ordering,
            reachability_: graph.reachability,
            core_distances_: graph.core_distances,
            labels_: labels,
            predecessors_: graph.predecessors,
            predecessor_: graph.predecessor,
            min_samples_: self.min_samples,
            cluster_hierarchy_,
        })
    }
}

impl<F: Float + Send + Sync + 'static> OPTICS<F> {
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

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Three tight 2-D clusters.
    fn three_blobs() -> Array2<f64> {
        Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1, 10.0, 0.0, 10.1, 0.0,
                10.0, 0.1,
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_ordering_covers_all_points() {
        let x = three_blobs();
        let fitted = OPTICS::<f64>::new(2).fit(&x, &()).unwrap();

        let mut sorted = fitted.ordering().to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..9).collect::<Vec<_>>());
    }

    #[test]
    fn test_reachability_length() {
        let x = three_blobs();
        let fitted = OPTICS::<f64>::new(2).fit(&x, &()).unwrap();
        assert_eq!(fitted.reachability().len(), 9);
    }

    #[test]
    fn test_core_distances_length() {
        let x = three_blobs();
        let fitted = OPTICS::<f64>::new(2).fit(&x, &()).unwrap();
        assert_eq!(fitted.core_distances().len(), 9);
    }

    #[test]
    fn test_labels_length() {
        let x = three_blobs();
        let fitted = OPTICS::<f64>::new(2).fit(&x, &()).unwrap();
        assert_eq!(fitted.labels().len(), 9);
    }

    #[test]
    fn test_core_points_have_finite_core_distance() {
        let x = three_blobs();
        // With min_samples=2 all tight-cluster points should be core points.
        let fitted = OPTICS::<f64>::new(2).fit(&x, &()).unwrap();
        // Points 0-8 each have at least one close neighbour.
        for i in 0..9 {
            // Core distance is finite because each point has neighbours.
            assert!(
                fitted.core_distances()[i].is_finite(),
                "expected finite core distance for point {i}"
            );
        }
    }

    #[test]
    fn test_isolated_point_infinite_core_distance() {
        // Add an isolated point far from the clusters.
        let mut data = three_blobs().into_raw_vec_and_offset().0;
        data.extend_from_slice(&[100.0, 100.0]);
        let x = Array2::from_shape_vec((10, 2), data).unwrap();

        // With max_eps=2.0, the isolated point has no neighbours, so its core
        // distance must be infinite regardless of min_samples.
        let fitted = OPTICS::<f64>::new(3)
            .with_max_eps(2.0)
            .fit(&x, &())
            .unwrap();
        assert!(
            fitted.core_distances()[9].is_infinite(),
            "isolated point should have infinite core distance"
        );
    }

    #[test]
    fn test_reachability_first_point_infinite() {
        let x = three_blobs();
        let fitted = OPTICS::<f64>::new(2).fit(&x, &()).unwrap();
        let first = fitted.ordering()[0];
        assert!(
            fitted.reachability()[first].is_infinite(),
            "first point in ordering should have infinite reachability"
        );
    }

    #[test]
    fn test_extract_clusters_valid_xi() {
        let x = three_blobs();
        let fitted = OPTICS::<f64>::new(2).fit(&x, &()).unwrap();
        let labels = fitted.extract_clusters(0.05).unwrap();
        assert_eq!(labels.len(), 9);
    }

    #[test]
    fn test_extract_clusters_invalid_xi_zero() {
        let x = three_blobs();
        let fitted = OPTICS::<f64>::new(2).fit(&x, &()).unwrap();
        assert!(fitted.extract_clusters(0.0).is_err());
    }

    #[test]
    fn test_extract_clusters_invalid_xi_one() {
        let x = three_blobs();
        let fitted = OPTICS::<f64>::new(2).fit(&x, &()).unwrap();
        assert!(fitted.extract_clusters(1.0).is_err());
    }

    #[test]
    fn test_invalid_min_samples_zero() {
        let x = three_blobs();
        let result = OPTICS::<f64>::new(0).fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_min_samples_one() {
        // min_samples=1 is now rejected (sklearn requires >= 2).
        let x = three_blobs();
        let result = OPTICS::<f64>::new(1).fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_max_eps_zero_accepted() {
        // sklearn accepts max_eps=0 (closed at 0); the fit runs and produces an
        // all-inf / all-noise degenerate result.
        let x = three_blobs();
        let result = OPTICS::<f64>::new(2).with_max_eps(0.0).fit(&x, &());
        assert!(result.is_ok(), "max_eps=0 should be accepted");
        if let Ok(fitted) = result {
            assert!(fitted.core_distances().iter().all(|c| c.is_infinite()));
            assert!(fitted.reachability().iter().all(|r| r.is_infinite()));
        }
    }

    #[test]
    fn test_invalid_max_eps_negative() {
        let x = three_blobs();
        let result = OPTICS::<f64>::new(2).with_max_eps(-1.0).fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_xi_zero_accepted() {
        // sklearn accepts xi=0 (closed at 0).
        let x = three_blobs();
        let result = OPTICS::<f64>::new(2).with_xi(0.0).fit(&x, &());
        assert!(result.is_ok());
    }

    #[test]
    fn test_xi_one_xi_method_errors() {
        // sklearn accepts xi=1 at validation but the Xi method raises a runtime
        // ZeroDivisionError (1-xi==0); the observable contract is "fit raises",
        // so ferrolearn errors on the Xi path (#2197), no panic.
        let x = three_blobs();
        let result = OPTICS::<f64>::new(2).with_xi(1.0).fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_xi_negative() {
        let x = three_blobs();
        let result = OPTICS::<f64>::new(2).with_xi(-0.1).fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_xi_above_one() {
        let x = three_blobs();
        let result = OPTICS::<f64>::new(2).with_xi(1.1).fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_data_error() {
        let x = Array2::<f64>::zeros((0, 2));
        let result = OPTICS::<f64>::new(2).fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_single_sample_rejected() {
        // A single sample cannot be fit by OPTICS: sklearn requires
        // `2 <= min_samples <= n_samples`, but n_samples=1 leaves no valid
        // min_samples, so `_validate_size` raises (min_samples=2 > 1). ferrolearn
        // mirrors that with an Err (#2197).
        let x: Array2<f64> = ndarray::array![[5.0, 5.0]];
        let result = OPTICS::<f64>::new(2).fit(&x, &());
        assert!(
            result.is_err(),
            "single-sample fit must error (min_samples=2 > n_samples=1)"
        );
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

        let fitted = OPTICS::<f32>::new(2).fit(&x, &()).unwrap();
        assert_eq!(fitted.ordering().len(), 6);
    }

    #[test]
    fn test_n_clusters_non_negative() {
        let x = three_blobs();
        let fitted = OPTICS::<f64>::new(2).fit(&x, &()).unwrap();
        // n_clusters() just counts distinct non-noise labels.
        let _ = fitted.n_clusters(); // Should not panic.
    }

    #[test]
    fn test_ordering_unique_indices() {
        let x = three_blobs();
        let fitted = OPTICS::<f64>::new(2).fit(&x, &()).unwrap();
        let ordering = fitted.ordering();
        let mut seen = std::collections::HashSet::new();
        for &idx in ordering {
            assert!(seen.insert(idx), "duplicate index {idx} in ordering");
        }
    }

    #[test]
    fn test_with_max_eps_limits_reachability() {
        let x = three_blobs();
        let max_eps = 0.5;
        let fitted = OPTICS::<f64>::new(2)
            .with_max_eps(max_eps)
            .fit(&x, &())
            .unwrap();
        // All finite reachability values must be <= max_eps.
        for &r in fitted.reachability() {
            if r.is_finite() {
                assert!(r <= max_eps + 1e-10);
            }
        }
    }

    #[test]
    fn test_predecessors_length() {
        let x = three_blobs();
        let fitted = OPTICS::<f64>::new(2).fit(&x, &()).unwrap();
        assert_eq!(fitted.predecessors().len(), 9);
    }

    #[test]
    fn test_first_point_has_no_predecessor() {
        let x = three_blobs();
        let fitted = OPTICS::<f64>::new(2).fit(&x, &()).unwrap();
        let first = fitted.ordering()[0];
        assert!(
            fitted.predecessors()[first].is_none(),
            "first point in ordering should have no predecessor"
        );
    }

    #[test]
    fn test_min_cluster_size_filters_small_clusters() {
        // Create data where one "cluster" is just 2 points and others are larger.
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.05, 0.05, // cluster of 4
                10.0, 10.0, 10.05, 10.0, // cluster of 2
                20.0, 20.0, 20.05, 20.0, // cluster of 2
            ],
        )
        .unwrap();

        let fitted = OPTICS::<f64>::new(2)
            .with_min_cluster_size(3)
            .fit(&x, &())
            .unwrap();

        // The small clusters (size 2) should be filtered out as noise.
        for &l in fitted.labels() {
            if l >= 0 {
                // Count how many points share this label.
                let count = fitted.labels().iter().filter(|&&c| c == l).count();
                assert!(
                    count >= 3,
                    "cluster with label {l} has only {count} points, expected >= 3"
                );
            }
        }
    }

    #[test]
    fn test_with_min_cluster_size_builder() {
        let optics = OPTICS::<f64>::new(5).with_min_cluster_size(10);
        assert_eq!(optics.min_cluster_size, Some(10));
    }
}
