//! Feature Agglomeration — hierarchical clustering of features.
//!
//! [`FeatureAgglomeration`] transposes the data and applies agglomerative
//! clustering to the *features* (columns) rather than the samples (rows).
//! After fitting, features within each cluster are pooled (by mean or max)
//! to produce a reduced-dimensionality representation.
//!
//! This is useful for supervised dimensionality reduction: correlated
//! features are grouped, and the group summary (e.g. mean) replaces
//! the original features.
//!
//! # Algorithm
//!
//! 1. **Fit**: transpose `X`, run agglomerative clustering on the columns
//!    (treated as data points) to obtain `n_clusters` feature groups.
//! 2. **Transform**: for each feature group, apply the pooling function
//!    (mean or max) across the grouped columns.  The output has
//!    `n_clusters` columns.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_cluster::{FeatureAgglomeration, PoolingFunc};
//! use ferrolearn_core::traits::{Fit, Transform};
//! use ndarray::Array2;
//!
//! let x = Array2::from_shape_vec((4, 6), vec![
//!     1.0, 1.1, 5.0, 5.1, 9.0, 9.1,
//!     2.0, 2.1, 6.0, 6.1, 8.0, 8.1,
//!     3.0, 3.1, 7.0, 7.1, 7.0, 7.1,
//!     4.0, 4.1, 8.0, 8.1, 6.0, 6.1,
//! ]).unwrap();
//!
//! let fa = FeatureAgglomeration::<f64>::new(3);
//! let fitted = fa.fit(&x, &()).unwrap();
//! let reduced = fitted.transform(&x).unwrap();
//! assert_eq!(reduced.ncols(), 3);
//! ```
//!
//! # `## REQ status`
//!
//! Binary classification (R-DEFER-2): two states only — SHIPPED needs impl + a
//! non-test production consumer + green verification + symbol-anchor + sklearn
//! `file:line`; NOT-STARTED carries the open prereq blocker. **`FeatureAgglomeration`
//! has NO PyO3 binding** — `grep -rln FeatureAgglomeration ferrolearn-python/` is
//! EMPTY (no `_RsFeatureAgglomeration`, no `ferrolearn.FeatureAgglomeration`). The
//! non-test production consumer is therefore the crate re-export at the crate root
//! (`pub use feature_agglomeration::{AgglomerativeLinkage, FeatureAgglomeration,
//! FittedFeatureAgglomeration, PoolingFunc}` in `ferrolearn-cluster/src/lib.rs`),
//! exposing `fit` / `transform` / `labels()` (+ `feature_labels()` /
//! `children()` / `distances()` / `n_leaves()` / `n_connected_components()`).
//! **End-to-end VALUE parity is now achieved (post-#938):** the AgglomerativeClustering
//! unit ships bit-exact `_hc_cut` `labels_` (commit 3e001cf4b), and
//! `FeatureAgglomeration::fit` delegates to `AgglomerativeClustering::fit(X.T)`
//! (`_agglomerative.py:1339`), so the now-correct `labels_` flow through unchanged
//! (NO fit-side relabel: the inner `labels_` are stored verbatim into
//! `feature_labels_`). `feature_labels_`/`labels()` is integer-EXACT vs sklearn
//! `FeatureAgglomeration(...).fit(X).labels_` for all four linkages, k∈{2,3}; and
//! because `fn transform` groups by ASCENDING label index (it accumulates into
//! `result[[i, label]]`, column j = pool over features with `label == j`), the
//! `transform` output is now VALUE-EXACT and COLUMN-ORDERED vs sklearn (mean +
//! max), `_feature_agglomeration.py:51-64`. Fitted attrs `children_`/`distances_`/
//! `n_leaves_`/`n_connected_components_` are delegated from the inner fit over `X.T`.
//! Green verification = the in-tree `feature_agglom` lib tests plus the live-sklearn
//! pins/guards: `divergence_feature_agglom_min_features_two` (#944) +
//! `green_feature_agglom_*` (`tests/divergence_feature_agglomeration.rs`) and the new
//! VALUE tests (`tests/divergence_feature_agglomeration_value.rs`):
//! `value_labels_exact_all_linkages_k2_k3`, `value_transform_mean_column_ordered`,
//! `value_transform_max_column_ordered`, `value_fitted_attrs_delegated`. Cites use
//! symbol anchors (ferrolearn) / `file:line` (sklearn 1.5.2, commit 156ef14). Live
//! oracle = installed sklearn 1.5.2. (REQ numbering follows
//! `.design/cluster/feature_agglomeration.md`.)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-4 (validation guards: `ensure_min_features=2`; `n_clusters >= 1`; `n_features >= n_clusters`; non-empty `X` — SHAPE/validation contract only) | SHIPPED (validation + shape + pooling-as-set contract only) | `fn fit` for `FeatureAgglomeration` rejects a 1-feature `X` with `FerroError::InvalidParameter { name: "X", reason: "Found array with {n} feature(s) while a minimum of 2 is required by FeatureAgglomeration" }`, mirroring `self._validate_data(X, ensure_min_features=2)` (`_agglomerative.py:1338`); rejects `n_clusters == 0` per `Interval(Integral, 1, None, closed="left")` (`:1281`); rejects `n_features < n_clusters` ("Cannot extract more clusters than samples", from `_hc_cut`); rejects empty `X` with `FerroError::InsufficientSamples`. The `Transform` impl (`fn transform`) returns output of SHAPE `(n_samples, n_clusters)`. Non-test consumer: crate re-export of `fit` / `transform` / `labels()` (`lib.rs`). Verified: now-PASSING pin `divergence_feature_agglom_min_features_two` (#944): `new(1).fit(X_(4,1))` returns `Err` (sklearn `FeatureAgglomeration(n_clusters=1).fit(X_1col)` raises `ValueError`); green guards `green_feature_agglom_transform_shape`, `green_feature_agglom_mean_pooling_as_set`, `green_feature_agglom_n_clusters_zero_rejected`, `green_feature_agglom_too_many_clusters_rejected`. GAP (NOT-STARTED): the sklearn `ValueError`/`InvalidParameterError` error ABI is NOT matched — ferrolearn uses `FerroError`. |
//! | REQ-1 (`transform` mean-pooling VALUE) | SHIPPED | `fn transform` (`PoolingFunc::Mean`) computes per-cluster mean (`sum/count`) into `result[[i, label]]`, column j ordered by ASCENDING label index, mirroring the `bincount` fast path of `AgglomerationTransform.transform` (`_feature_agglomeration.py:51-57`: `nX[i] = np.bincount(labels_, X[i,:]) / size`, column = label index 0..n_clusters-1). With the now-correct delegated `labels_` (#938 shipped, commit 3e001cf4b), the output is VALUE-EXACT and COLUMN-ORDERED vs sklearn. Non-test consumer: crate re-export (`lib.rs`); also consumed by `api_proof.rs`/`conformance_wave4.rs`. Verified: `value_transform_mean_column_ordered` — ferrolearn `transform(X)` == sklearn `FeatureAgglomeration(n_clusters=3, pooling_func=np.mean).fit(X).transform(X)` (full matrix, ~1e-9) for ward+single; sklearn row0 = `[1.05, 9.05, 5.05]`, ferrolearn row0 = `[1.05, 9.05, 5.05]`. |
//! | REQ-2 (`transform` max-pooling VALUE) | SHIPPED | `fn transform` (`PoolingFunc::Max`) takes per-cluster max into `result[[i, label]]`, ascending label-index column order, mirroring the general pooling path (`_feature_agglomeration.py:58-63`: `[pooling_func(X[:, labels_==l], axis=1) for l in np.unique(labels_)]`, columns by sorted unique label = 0..n_clusters-1). VALUE-EXACT and COLUMN-ORDERED post-#938. Non-test consumer: crate re-export (`lib.rs`). Verified: `value_transform_max_column_ordered` — full-matrix equality vs sklearn `pooling_func=np.max` (ward+complete); sklearn row0 = `[1.1, 9.1, 5.1]`, ferrolearn row0 = `[1.1, 9.1, 5.1]`. |
//! | REQ-3 (`labels_` feature-cluster VALUE parity) | SHIPPED | `fn fit` delegates to `AgglomerativeClustering::new(n_clusters).with_linkage(...)` on `X.T` and stores `fitted_agg.labels_` verbatim into `feature_labels_` (NO post-relabel), mirroring `super()._fit(X.T)` (`_agglomerative.py:1339`). The AgglomerativeClustering unit now ships bit-exact `_hc_cut` `labels_` (`agglomerative.rs fn hc_cut`, `:731-775`; `np.searchsorted` numbering `:1105`; commit 3e001cf4b), so `feature_labels_`/`labels()` is integer-EXACT: sklearn `[0,0,2,2,1,1]` (k=3) / `[1,1,0,0,0,0]` (k=2) == ferrolearn. Accessor `fn labels` (sklearn name) returns the same data as `fn feature_labels`. Non-test consumer: crate re-export (`lib.rs`). Verified: `value_labels_exact_all_linkages_k2_k3` — integer-exact for {ward,complete,average,single} × {2,3}. |
//! | REQ-5 (linkage variants ward/complete/average/single) | SHIPPED | `fn map_linkage` maps all four `AgglomerativeLinkage` variants to `Linkage`, delegated to `AgglomerativeClustering`, mirroring `_TREE_BUILDERS` (`_agglomerative.py:720-725`/`:1290`). VALUE parity (labels_ + transform) holds across all four post-#938. Non-test consumer: crate re-export (`lib.rs`). Verified: `value_labels_exact_all_linkages_k2_k3` (all four linkages) + `value_transform_mean_column_ordered` (ward+single) + `value_transform_max_column_ordered` (ward+complete). |
//! | REQ-6 (`n_clusters=2` default + missing params metric/memory/connectivity/compute_full_tree/distance_threshold/compute_distances) | NOT-STARTED | open prereq blocker **#941**. sklearn `__init__` (`_agglomerative.py:1296-1319`) takes 9 params with `n_clusters=2` default. `FeatureAgglomeration<F>` REQUIRES `n_clusters` (`fn new`, no default) and has only `linkage`/`pooling_func`. `distance_threshold` (cut by distance with `n_clusters=None`, `:1281`/`:1091-1092`) is materially absent. |
//! | REQ-7 (`pooling_func` as arbitrary callable) | NOT-STARTED | open prereq blocker **#941**. sklearn `_parameter_constraints["pooling_func"] = [callable]` (`_agglomerative.py:1291`) accepts any callable (default `np.mean`, `:1305`). ferrolearn offers only the closed `PoolingFunc::{Mean, Max}` enum (`fn with_pooling_func`). |
//! | REQ-8 (`inverse_transform`) | SHIPPED | `FittedFeatureAgglomeration::inverse_transform` broadcasts each cluster's pooled value back to its member features: `result[i, f] = xred[i, labels_[f]]`, mirroring sklearn `AgglomerationTransform.inverse_transform` (`_feature_agglomeration.py:66-92`: `unil, inverse = np.unique(labels_, return_inverse=True); return X[..., inverse]` — `inverse == labels_` for the contiguous `_hc_cut` numbering). Output shape `(n_samples, n_features)`; accepts `xred.ncols() >= n_clusters` (ignoring trailing extra columns, matching sklearn's numpy fancy-index `X[..., inverse]`, #2187) and rejects `xred.ncols() < n_clusters` with `FerroError::ShapeMismatch` (sklearn raises `IndexError`). Non-test consumer: crate re-export (`lib.rs`). Verified: `value_inverse_transform_roundtrip_and_broadcast` (full-matrix ~1e-12, all four linkages) + `divergence_inverse_transform_extra_cols_ignored` (#2187: `(2,4)` xred with n_clusters=3 ignores the 4th column, matching sklearn). |
//! | REQ-9 (fitted attrs labels_/n_leaves_/n_connected_components_/children_/distances_) | SHIPPED | `fn fit` stores the inner `AgglomerativeClustering::fit(X.T)` attributes into `FittedFeatureAgglomeration` (`children_`, `distances_`, `n_leaves_`, `n_connected_components_`), delegating exactly as sklearn `FeatureAgglomeration._fit` → `AgglomerativeClustering._fit(X.T)` (`_agglomerative.py:1339`, sets `labels_`/`n_clusters_`/`n_leaves_`/`n_connected_components_`/`children_`/`distances_` at `:1083-1095`). Accessors: `fn labels` (sklearn name, alias of `feature_labels`), `fn children`, `fn distances` (`Option`, `Some` iff `with_compute_distances(true)` — mirrors sklearn `compute_distances`, `:1319`), `fn n_leaves` (= `n_features`), `fn n_connected_components` (= 1 for the unstructured path). `with_compute_distances` passthrough sets the inner `compute_distances`. Non-test consumer: crate re-export (`lib.rs`). Verified: `value_fitted_attrs_delegated` — `children_` == sklearn `FeatureAgglomeration(compute_distances=True).fit(X).children_`; `distances_` == sklearn `distances_` (~1e-9); `n_leaves_ == 6 == n_features`; `n_connected_components_ == 1`. |
//! | REQ-10 (PyO3 binding) | NOT-STARTED | open prereq blocker **#943**. `grep -rln FeatureAgglomeration ferrolearn-python/` is EMPTY — no `_RsFeatureAgglomeration`, so `import ferrolearn` cannot reach `FeatureAgglomeration`. The only non-test consumer of `fit` / `transform` / `feature_labels()` is the crate re-export (`lib.rs`). |
//! | REQ-11 (ferray substrate) | NOT-STARTED | open prereq blocker **#943**. `feature_agglomeration.rs` imports `ndarray::{Array1, Array2}` + `num_traits::Float`; the delegated `agglomerative.rs` is likewise on `ndarray`. Not migrated to `ferray-core` / `ferray::linalg` (R-SUBSTRATE-1/2). |

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Transform};
use ndarray::{Array1, Array2};
use num_traits::Float;

use crate::agglomerative::{AgglomerativeClustering, Linkage};

// ─────────────────────────────────────────────────────────────────────────────
// Public enums
// ─────────────────────────────────────────────────────────────────────────────

/// The linkage criterion used by [`FeatureAgglomeration`].
///
/// Re-uses the same linkage strategies as [`AgglomerativeClustering`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgglomerativeLinkage {
    /// Ward linkage (minimises within-cluster variance).
    Ward,
    /// Complete linkage (max pairwise distance).
    Complete,
    /// Average linkage (mean pairwise distance).
    Average,
    /// Single linkage (min pairwise distance).
    Single,
}

/// The pooling function applied to grouped features during transformation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolingFunc {
    /// Take the mean of features in each cluster.
    Mean,
    /// Take the maximum of features in each cluster.
    Max,
}

// ─────────────────────────────────────────────────────────────────────────────
// FeatureAgglomeration (unfitted)
// ─────────────────────────────────────────────────────────────────────────────

/// Feature Agglomeration configuration (unfitted).
///
/// Call [`Fit::fit`] to cluster the features and obtain a
/// [`FittedFeatureAgglomeration`].
///
/// # Type Parameters
///
/// - `F`: floating-point scalar type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct FeatureAgglomeration<F> {
    /// Target number of feature clusters.
    n_clusters: usize,
    /// Linkage strategy for the agglomerative clustering of features.
    linkage: AgglomerativeLinkage,
    /// Pooling function applied during transformation.
    pooling_func: PoolingFunc,
    /// Whether to compute and store the per-merge `distances_` of the inner
    /// dendrogram, mirroring sklearn `compute_distances` (default `false`,
    /// `_agglomerative.py:1319`).
    compute_distances: bool,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> FeatureAgglomeration<F> {
    /// Create a new `FeatureAgglomeration` that reduces to `n_clusters` features.
    ///
    /// Defaults: `linkage = Ward`, `pooling_func = Mean`, `compute_distances = false`.
    #[must_use]
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            linkage: AgglomerativeLinkage::Ward,
            pooling_func: PoolingFunc::Mean,
            compute_distances: false,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the linkage criterion.
    #[must_use]
    pub fn with_linkage(mut self, linkage: AgglomerativeLinkage) -> Self {
        self.linkage = linkage;
        self
    }

    /// Set the pooling function.
    #[must_use]
    pub fn with_pooling_func(mut self, pooling: PoolingFunc) -> Self {
        self.pooling_func = pooling;
        self
    }

    /// Compute and store the per-merge linkage `distances_` of the inner
    /// dendrogram (over the transposed feature matrix), mirroring sklearn
    /// `FeatureAgglomeration(compute_distances=True)` (`_agglomerative.py:1319`).
    ///
    /// When set, [`FittedFeatureAgglomeration::distances`] returns `Some`.
    #[must_use]
    pub fn with_compute_distances(mut self, compute_distances: bool) -> Self {
        self.compute_distances = compute_distances;
        self
    }

    /// Return the configured number of feature clusters.
    #[must_use]
    pub fn n_clusters(&self) -> usize {
        self.n_clusters
    }

    /// Return the configured linkage.
    #[must_use]
    pub fn linkage(&self) -> AgglomerativeLinkage {
        self.linkage
    }

    /// Return the configured pooling function.
    #[must_use]
    pub fn pooling_func(&self) -> PoolingFunc {
        self.pooling_func
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FittedFeatureAgglomeration
// ─────────────────────────────────────────────────────────────────────────────

/// Fitted Feature Agglomeration model.
///
/// Stores the cluster label for each original feature and the pooling
/// strategy.  Implements [`Transform`] to reduce the dimensionality of
/// new data.
#[derive(Debug, Clone)]
pub struct FittedFeatureAgglomeration<F> {
    /// Cluster label for each original feature, length `n_features`.
    ///
    /// This is sklearn's `labels_` attribute: it is exactly the inner
    /// [`AgglomerativeClustering`]'s `labels_` over the transposed feature
    /// matrix `X.T` (`_agglomerative.py:1339`, `FeatureAgglomeration._fit`
    /// → `AgglomerativeClustering._fit(X.T)`).
    feature_labels_: Array1<usize>,
    /// Number of feature clusters.
    n_clusters_: usize,
    /// Number of original features.
    n_features_: usize,
    /// Pooling function to aggregate features within each cluster.
    pooling_func_: PoolingFunc,
    /// The merge history (dendrogram) of the inner clustering over `X.T`,
    /// length `n_features - 1`. Mirrors sklearn `children_`
    /// (`_agglomerative.py:1083-1095`, delegated from `AgglomerativeClustering`).
    children_: Vec<(usize, usize)>,
    /// The per-merge linkage distances of the inner clustering over `X.T`,
    /// in `children_` row order. `Some` when `compute_distances` was set.
    /// Mirrors sklearn `distances_`.
    distances_: Option<Array1<F>>,
    /// Number of leaves in the inner hierarchical tree (`== n_features` for the
    /// unstructured path). Mirrors sklearn `n_leaves_`.
    n_leaves_: usize,
    /// Estimated number of connected components in the inner graph (always `1`
    /// for the unstructured `connectivity=None` path). Mirrors sklearn
    /// `n_connected_components_`.
    n_connected_components_: usize,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> FittedFeatureAgglomeration<F> {
    /// Return the cluster label assigned to each feature.
    ///
    /// This is the sklearn name for the per-feature cluster assignment
    /// (`labels_`). Identical data to [`feature_labels`](Self::feature_labels).
    #[must_use]
    pub fn labels(&self) -> &Array1<usize> {
        &self.feature_labels_
    }

    /// Return the cluster label assigned to each feature.
    ///
    /// Alias of [`labels`](Self::labels) retained for backward compatibility;
    /// sklearn names this attribute `labels_`.
    #[must_use]
    pub fn feature_labels(&self) -> &Array1<usize> {
        &self.feature_labels_
    }

    /// Return the number of feature clusters.
    #[must_use]
    pub fn n_clusters(&self) -> usize {
        self.n_clusters_
    }

    /// Return the number of original features.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.n_features_
    }

    /// Return the merge history (dendrogram) of the inner clustering over `X.T`.
    ///
    /// Mirrors sklearn's `children_`, length `n_features - 1`. Delegated from
    /// the inner [`AgglomerativeClustering`] fit (`_agglomerative.py:1339`).
    #[must_use]
    pub fn children(&self) -> &[(usize, usize)] {
        &self.children_
    }

    /// Return the per-merge linkage distances of the inner dendrogram (in
    /// `children_` row order), or `None` if `compute_distances` was not set.
    ///
    /// Mirrors sklearn's `distances_` attribute.
    #[must_use]
    pub fn distances(&self) -> Option<&Array1<F>> {
        self.distances_.as_ref()
    }

    /// Return the number of leaves in the inner hierarchical tree
    /// (`== n_features` for the unstructured path). Mirrors sklearn `n_leaves_`.
    #[must_use]
    pub fn n_leaves(&self) -> usize {
        self.n_leaves_
    }

    /// Return the estimated number of connected components in the inner graph
    /// (always `1` for the unstructured `connectivity=None` path). Mirrors
    /// sklearn `n_connected_components_`.
    #[must_use]
    pub fn n_connected_components(&self) -> usize {
        self.n_connected_components_
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Map between linkage types
// ─────────────────────────────────────────────────────────────────────────────

/// Convert [`AgglomerativeLinkage`] to the internal [`Linkage`] enum.
fn map_linkage(l: AgglomerativeLinkage) -> Linkage {
    match l {
        AgglomerativeLinkage::Ward => Linkage::Ward,
        AgglomerativeLinkage::Complete => Linkage::Complete,
        AgglomerativeLinkage::Average => Linkage::Average,
        AgglomerativeLinkage::Single => Linkage::Single,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Trait impls
// ─────────────────────────────────────────────────────────────────────────────

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for FeatureAgglomeration<F> {
    type Fitted = FittedFeatureAgglomeration<F>;
    type Error = FerroError;

    /// Fit feature agglomeration by clustering the features (columns).
    ///
    /// Transposes `x` so each column becomes a row (data point) and
    /// runs agglomerative clustering.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `n_clusters == 0`.
    /// - [`FerroError::InsufficientSamples`] if `n_features < n_clusters`.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedFeatureAgglomeration<F>, FerroError> {
        let n_features = x.ncols();

        if self.n_clusters == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_clusters".into(),
                reason: "must be at least 1".into(),
            });
        }
        if n_features < 2 {
            return Err(FerroError::InvalidParameter {
                name: "X".into(),
                reason: format!(
                    "Found array with {n_features} feature(s) while a minimum of 2 is required by FeatureAgglomeration"
                ),
            });
        }
        if n_features < self.n_clusters {
            return Err(FerroError::InvalidParameter {
                name: "n_clusters".into(),
                reason: format!(
                    "n_clusters ({}) exceeds n_features ({})",
                    self.n_clusters, n_features
                ),
            });
        }
        if x.nrows() == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "FeatureAgglomeration::fit requires at least 1 sample".into(),
            });
        }

        // Transpose: each feature (column) becomes a row.
        // Use as_standard_layout() to ensure row-major (C) order, which
        // is required for AgglomerativeClustering's internal as_slice().
        let x_t = x.t().as_standard_layout().into_owned();

        // Run agglomerative clustering on the transposed data.
        //
        // sklearn `FeatureAgglomeration._fit` delegates verbatim to
        // `AgglomerativeClustering._fit(X.T)` (`_agglomerative.py:1339`), so
        // `labels_`/`children_`/`distances_`/`n_leaves_`/`n_connected_components_`
        // are exactly the inner estimator's attributes over `X.T`.
        let agg = AgglomerativeClustering::<F>::new(self.n_clusters)
            .with_linkage(map_linkage(self.linkage))
            .with_compute_distances(self.compute_distances);
        let fitted_agg = agg.fit(&x_t, &())?;

        Ok(FittedFeatureAgglomeration {
            feature_labels_: fitted_agg.labels_,
            n_clusters_: self.n_clusters,
            n_features_: n_features,
            pooling_func_: self.pooling_func,
            children_: fitted_agg.children_,
            distances_: fitted_agg.distances_,
            n_leaves_: fitted_agg.n_leaves_,
            n_connected_components_: fitted_agg.n_connected_components_,
            _marker: std::marker::PhantomData,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedFeatureAgglomeration<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Transform data by pooling features within each cluster.
    ///
    /// The output has shape `(n_samples, n_clusters)`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns does not
    /// match the number of features seen during fitting.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        if x.ncols() != self.n_features_ {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), self.n_features_],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedFeatureAgglomeration::transform".into(),
            });
        }

        let n_samples = x.nrows();
        let mut result = Array2::<F>::zeros((n_samples, self.n_clusters_));

        match self.pooling_func_ {
            PoolingFunc::Mean => {
                // Count features per cluster.
                let mut counts = vec![0usize; self.n_clusters_];
                for &label in &self.feature_labels_ {
                    counts[label] += 1;
                }

                // Sum features per cluster.
                for i in 0..n_samples {
                    for (j, &label) in self.feature_labels_.iter().enumerate() {
                        result[[i, label]] = result[[i, label]] + x[[i, j]];
                    }
                }

                // Divide by count to get mean.
                for i in 0..n_samples {
                    for c in 0..self.n_clusters_ {
                        if counts[c] > 0 {
                            result[[i, c]] = result[[i, c]] / F::from(counts[c]).unwrap();
                        }
                    }
                }
            }
            PoolingFunc::Max => {
                // Initialize with negative infinity.
                result.fill(F::neg_infinity());

                for i in 0..n_samples {
                    for (j, &label) in self.feature_labels_.iter().enumerate() {
                        if x[[i, j]] > result[[i, label]] {
                            result[[i, label]] = x[[i, j]];
                        }
                    }
                }
            }
        }

        Ok(result)
    }
}

impl<F: Float + Send + Sync + 'static> FittedFeatureAgglomeration<F> {
    /// Inverse the pooling transformation: broadcast each cluster's pooled
    /// value back to every feature in that cluster.
    ///
    /// Mirrors sklearn `AgglomerationTransform.inverse_transform`
    /// (`_feature_agglomeration.py:66-92`): `unil, inverse =
    /// np.unique(labels_, return_inverse=True); return X[..., inverse]`. Since
    /// `labels_` is the contiguous `_hc_cut` numbering `0..n_clusters`,
    /// `inverse == labels_`, so output column `f` is reduced column
    /// `labels_[f]`: `result[i, f] = xred[i, labels_[f]]`.
    ///
    /// `xred` has shape `(n_samples, n_clusters)`; the result has shape
    /// `(n_samples, n_features)`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `xred.ncols() < n_clusters`.
    /// Mirroring sklearn's numpy fancy-index `X[..., inverse]` (which reads only
    /// columns `0..n_clusters` and ignores any trailing columns, #2187), a wider
    /// `xred` is accepted and the extra columns are ignored; too few columns is
    /// the only error (sklearn raises `IndexError`).
    pub fn inverse_transform(&self, xred: &Array2<F>) -> Result<Array2<F>, FerroError> {
        if xred.ncols() < self.n_clusters_ {
            return Err(FerroError::ShapeMismatch {
                expected: vec![xred.nrows(), self.n_clusters_],
                actual: vec![xred.nrows(), xred.ncols()],
                context: "FittedFeatureAgglomeration::inverse_transform".into(),
            });
        }

        let n_samples = xred.nrows();
        let mut result = Array2::<F>::zeros((n_samples, self.n_features_));
        for i in 0..n_samples {
            for (f, &label) in self.feature_labels_.iter().enumerate() {
                result[[i, f]] = xred[[i, label]];
            }
        }
        Ok(result)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn make_correlated_features() -> Array2<f64> {
        // 6 features: (0,1) correlated, (2,3) correlated, (4,5) correlated.
        Array2::from_shape_vec(
            (5, 6),
            vec![
                1.0, 1.1, 5.0, 5.1, 9.0, 9.1, 2.0, 2.1, 6.0, 6.1, 8.0, 8.1, 3.0, 3.1, 7.0, 7.1,
                7.0, 7.1, 4.0, 4.1, 8.0, 8.1, 6.0, 6.1, 5.0, 5.1, 9.0, 9.1, 5.0, 5.1,
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_feature_agglom_basic() {
        let x = make_correlated_features();
        let fa = FeatureAgglomeration::<f64>::new(3);
        let fitted = fa.fit(&x, &()).unwrap();
        let reduced = fitted.transform(&x).unwrap();
        assert_eq!(reduced.dim(), (5, 3));
    }

    #[test]
    fn test_feature_agglom_output_shape() {
        let x = make_correlated_features();
        let fa = FeatureAgglomeration::<f64>::new(2);
        let fitted = fa.fit(&x, &()).unwrap();
        let reduced = fitted.transform(&x).unwrap();
        assert_eq!(reduced.ncols(), 2);
        assert_eq!(reduced.nrows(), 5);
    }

    #[test]
    fn test_feature_agglom_labels_valid_range() {
        let x = make_correlated_features();
        let fa = FeatureAgglomeration::<f64>::new(3);
        let fitted = fa.fit(&x, &()).unwrap();
        for &l in fitted.feature_labels() {
            assert!(l < 3, "label {l} out of range");
        }
    }

    #[test]
    fn test_feature_agglom_correlated_grouped() {
        // With 4 features merging to 2 clusters, pairs of nearly-identical
        // features should end up together.  Use Single linkage to guarantee
        // nearest-neighbor pairing.
        let x = Array2::from_shape_vec(
            (5, 4),
            vec![
                // feat 0  feat 1  feat 2   feat 3
                1.0, 1.001, 100.0, 100.001, 2.0, 2.001, 90.0, 90.001, 3.0, 3.001, 80.0, 80.001, 4.0,
                4.001, 70.0, 70.001, 5.0, 5.001, 60.0, 60.001,
            ],
        )
        .unwrap();
        let fa = FeatureAgglomeration::<f64>::new(2).with_linkage(AgglomerativeLinkage::Single);
        let fitted = fa.fit(&x, &()).unwrap();
        let labels = fitted.feature_labels();
        // Features 0 and 1 are nearly identical, should be in the same cluster.
        assert_eq!(labels[0], labels[1]);
        // Features 2 and 3 are nearly identical.
        assert_eq!(labels[2], labels[3]);
        // The two pairs should be in different clusters.
        assert_ne!(labels[0], labels[2]);
    }

    #[test]
    fn test_feature_agglom_mean_pooling() {
        // Simple case: two features that should be grouped.
        let x = Array2::from_shape_vec((3, 2), vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]).unwrap();
        let fa = FeatureAgglomeration::<f64>::new(1);
        let fitted = fa.fit(&x, &()).unwrap();
        let reduced = fitted.transform(&x).unwrap();
        assert_eq!(reduced.ncols(), 1);
        // Mean of (2, 4) = 3, (6, 8) = 7, (10, 12) = 11.
        assert_abs_diff_eq!(reduced[[0, 0]], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(reduced[[1, 0]], 7.0, epsilon = 1e-10);
        assert_abs_diff_eq!(reduced[[2, 0]], 11.0, epsilon = 1e-10);
    }

    #[test]
    fn test_feature_agglom_max_pooling() {
        let x = Array2::from_shape_vec((3, 2), vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]).unwrap();
        let fa = FeatureAgglomeration::<f64>::new(1).with_pooling_func(PoolingFunc::Max);
        let fitted = fa.fit(&x, &()).unwrap();
        let reduced = fitted.transform(&x).unwrap();
        assert_eq!(reduced.ncols(), 1);
        // Max of (2, 4) = 4, (6, 8) = 8, (10, 12) = 12.
        assert_abs_diff_eq!(reduced[[0, 0]], 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(reduced[[1, 0]], 8.0, epsilon = 1e-10);
        assert_abs_diff_eq!(reduced[[2, 0]], 12.0, epsilon = 1e-10);
    }

    #[test]
    fn test_feature_agglom_complete_linkage() {
        let x = make_correlated_features();
        let fa = FeatureAgglomeration::<f64>::new(3).with_linkage(AgglomerativeLinkage::Complete);
        let fitted = fa.fit(&x, &()).unwrap();
        let reduced = fitted.transform(&x).unwrap();
        assert_eq!(reduced.ncols(), 3);
    }

    #[test]
    fn test_feature_agglom_average_linkage() {
        let x = make_correlated_features();
        let fa = FeatureAgglomeration::<f64>::new(3).with_linkage(AgglomerativeLinkage::Average);
        let fitted = fa.fit(&x, &()).unwrap();
        let reduced = fitted.transform(&x).unwrap();
        assert_eq!(reduced.ncols(), 3);
    }

    #[test]
    fn test_feature_agglom_single_linkage() {
        let x = make_correlated_features();
        let fa = FeatureAgglomeration::<f64>::new(3).with_linkage(AgglomerativeLinkage::Single);
        let fitted = fa.fit(&x, &()).unwrap();
        let reduced = fitted.transform(&x).unwrap();
        assert_eq!(reduced.ncols(), 3);
    }

    #[test]
    fn test_feature_agglom_n_clusters_equals_n_features() {
        let x = make_correlated_features();
        let fa = FeatureAgglomeration::<f64>::new(6);
        let fitted = fa.fit(&x, &()).unwrap();
        let reduced = fitted.transform(&x).unwrap();
        // No reduction; each feature is its own cluster.
        assert_eq!(reduced.ncols(), 6);
    }

    #[test]
    fn test_feature_agglom_zero_clusters_error() {
        let x = make_correlated_features();
        let fa = FeatureAgglomeration::<f64>::new(0);
        assert!(fa.fit(&x, &()).is_err());
    }

    #[test]
    fn test_feature_agglom_too_many_clusters_error() {
        let x = make_correlated_features();
        let fa = FeatureAgglomeration::<f64>::new(10); // only 6 features
        assert!(fa.fit(&x, &()).is_err());
    }

    #[test]
    fn test_feature_agglom_empty_data_error() {
        let x = Array2::<f64>::zeros((0, 4));
        let fa = FeatureAgglomeration::<f64>::new(2);
        assert!(fa.fit(&x, &()).is_err());
    }

    #[test]
    fn test_feature_agglom_transform_shape_mismatch() {
        let x = make_correlated_features();
        let fa = FeatureAgglomeration::<f64>::new(3);
        let fitted = fa.fit(&x, &()).unwrap();
        let x_bad =
            Array2::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_feature_agglom_f32() {
        let x = Array2::<f32>::from_shape_vec(
            (4, 4),
            vec![
                1.0, 1.1, 5.0, 5.1, 2.0, 2.1, 6.0, 6.1, 3.0, 3.1, 7.0, 7.1, 4.0, 4.1, 8.0, 8.1,
            ],
        )
        .unwrap();
        let fa = FeatureAgglomeration::<f32>::new(2);
        let fitted = fa.fit(&x, &()).unwrap();
        let reduced = fitted.transform(&x).unwrap();
        assert_eq!(reduced.ncols(), 2);
    }

    #[test]
    fn test_feature_agglom_getters() {
        let fa = FeatureAgglomeration::<f64>::new(3)
            .with_linkage(AgglomerativeLinkage::Complete)
            .with_pooling_func(PoolingFunc::Max);
        assert_eq!(fa.n_clusters(), 3);
        assert_eq!(fa.linkage(), AgglomerativeLinkage::Complete);
        assert_eq!(fa.pooling_func(), PoolingFunc::Max);
    }

    #[test]
    fn test_feature_agglom_n_features_getter() {
        let x = make_correlated_features();
        let fa = FeatureAgglomeration::<f64>::new(3);
        let fitted = fa.fit(&x, &()).unwrap();
        assert_eq!(fitted.n_features(), 6);
        assert_eq!(fitted.n_clusters(), 3);
    }
}
