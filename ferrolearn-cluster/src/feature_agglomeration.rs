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
//! exposing `fit` / `transform` / `feature_labels()`. **Honest underclaim
//! (R-HONEST-3): this unit does NOT achieve `FeatureAgglomeration` end-to-end VALUE
//! parity** — the feature PARTITION matches sklearn (`{0,1}`/`{2,3}`/`{4,5}` on the
//! fixture), but the integer `labels_` index is PERMUTED (sklearn `_hc_cut`
//! `_agglomerative.py:1099` + `np.searchsorted(np.unique(labels), labels)` `:1105` →
//! `[0,0,2,2,1,1]`; ferrolearn relabels by `active`-slot order in
//! `agglomerative.rs::agglomerate` → `[0,0,1,1,2,2]`), which cascades into the
//! `transform` output COLUMN ORDER (sklearn orders columns by `np.unique(labels_)`,
//! `_feature_agglomeration.py:62`). The root cause is OWNED by the
//! AgglomerativeClustering unit (`agglomerative.rs`), so `labels_`/transform-VALUE
//! parity is NOT-STARTED here. The ONLY contracts that VALUE-match the live oracle
//! with a real consumer are the validation guards (`ensure_min_features=2`,
//! `n_clusters >= 1`, `n_features >= n_clusters`), the transform output SHAPE
//! `(n_samples, n_clusters)`, and the pooling ARITHMETIC as an unordered set.
//! Green verification = the in-tree `feature_agglom` lib tests plus the live-sklearn
//! pin/guards (`ferrolearn-cluster/tests/divergence_feature_agglomeration.rs`): the
//! now-PASSING pin `divergence_feature_agglom_min_features_two` (#944) and the green
//! guards `green_feature_agglom_transform_shape`,
//! `green_feature_agglom_mean_pooling_as_set`,
//! `green_feature_agglom_n_clusters_zero_rejected`,
//! `green_feature_agglom_too_many_clusters_rejected`. Cites use symbol anchors
//! (ferrolearn) / `file:line` (sklearn 1.5.2, commit 156ef14). Live oracle =
//! installed sklearn 1.5.2. (REQ numbering follows
//! `.design/cluster/feature_agglomeration.md`.)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-4 (validation guards: `ensure_min_features=2`; `n_clusters >= 1`; `n_features >= n_clusters`; non-empty `X` — SHAPE/validation contract only) | SHIPPED (validation + shape + pooling-as-set contract only) | `fn fit` for `FeatureAgglomeration` rejects a 1-feature `X` with `FerroError::InvalidParameter { name: "X", reason: "Found array with {n} feature(s) while a minimum of 2 is required by FeatureAgglomeration" }`, mirroring `self._validate_data(X, ensure_min_features=2)` (`_agglomerative.py:1338`); rejects `n_clusters == 0` per `Interval(Integral, 1, None, closed="left")` (`:1281`); rejects `n_features < n_clusters` ("Cannot extract more clusters than samples", from `_hc_cut`); rejects empty `X` with `FerroError::InsufficientSamples`. The `Transform` impl (`fn transform`) returns output of SHAPE `(n_samples, n_clusters)`. Non-test consumer: crate re-export of `fit` / `transform` / `feature_labels()` (`lib.rs`). Verified: now-PASSING pin `divergence_feature_agglom_min_features_two` (#944): `new(1).fit(X_(4,1))` returns `Err` (sklearn `FeatureAgglomeration(n_clusters=1).fit(X_1col)` raises `ValueError`); green guards `green_feature_agglom_transform_shape` (`transform(X).dim() == (5, 3)`), `green_feature_agglom_mean_pooling_as_set` (per-row pooled values match sklearn as a SORTED set), `green_feature_agglom_n_clusters_zero_rejected` (`new(0).fit` → `Err`), `green_feature_agglom_too_many_clusters_rejected` (`new(10).fit(X_6col)` → `Err`). GAP (NOT-STARTED): the sklearn `ValueError`/`InvalidParameterError` error ABI is NOT matched — ferrolearn uses `FerroError`. Column-ordered transform VALUE and `labels_` VALUE are NOT claimed here (REQ-1/2/3). |
//! | REQ-1 (`transform` mean-pooling VALUE) | NOT-STARTED | open prereq blocker **#938** (depends on REQ-3). `fn transform` (`PoolingFunc::Mean`) computes per-cluster mean (`sum/count`) into `(n_samples, n_clusters)`, mirroring the `bincount` fast path of `AgglomerationTransform.transform` (`_feature_agglomeration.py:51-57`). The pooling ARITHMETIC is correct (guard `green_feature_agglom_mean_pooling_as_set`), but the output COLUMN ORDER diverges from sklearn (which orders by `np.unique(labels_)`, `:62`) because the underlying label index is permuted (REQ-3): sklearn `transform(X)[0] = [1.05, 9.05, 5.05]` vs ferrolearn `[1.05, 5.05, 9.05]`. Equal only as an unordered column set. Unblocks once REQ-3 lands. |
//! | REQ-2 (`transform` max-pooling VALUE) | NOT-STARTED | open prereq blocker **#938** (depends on REQ-3). `fn transform` (`PoolingFunc::Max`) takes per-cluster max, mirroring the general pooling path (`_feature_agglomeration.py:58-63`, `np.max` callable). Per-cluster max is correct; column order diverges identically: sklearn `transform(X)[0] = [1.1, 9.1, 5.1]` vs ferrolearn `[1.1, 5.1, 9.1]`. Gated on REQ-3. |
//! | REQ-3 (`labels_` feature-cluster VALUE parity) | NOT-STARTED | open prereq blocker **#938** (owned by the `agglomerative.rs` unit; the ROOT cause). `fn fit` delegates to `AgglomerativeClustering::new(n_clusters).with_linkage(...)` on `X.T`, storing `feature_labels_`, mirroring `super()._fit(X.T)` (`_agglomerative.py:1339`). The PARTITION matches sklearn (`{0,1}`/`{2,3}`/`{4,5}`) but integer labels are PERMUTED: sklearn `_hc_cut` (`:1099`) + `np.searchsorted(np.unique(labels), labels)` (`:1105`) → `[0,0,2,2,1,1]`; ferrolearn `agglomerative.rs::agglomerate` relabels by `active`-slot order → `[0,0,1,1,2,2]`. Fixing the `_hc_cut`/`searchsorted` label-numbering in `agglomerative.rs` is the single root cause; REQ-1/REQ-2 unblock once it lands. |
//! | REQ-5 (linkage variants ward/complete/average/single) | NOT-STARTED | open prereq blocker **#938** (shares the REQ-3 root cause). `fn map_linkage` maps all four `AgglomerativeLinkage` variants to `Linkage`, delegated to `AgglomerativeClustering`, mirroring `_TREE_BUILDERS` (`_agglomerative.py:1290`). The partition matches across all four on the fixture, but `labels_`/transform VALUE is gated on REQ-3. |
//! | REQ-6 (`n_clusters=2` default + missing params metric/memory/connectivity/compute_full_tree/distance_threshold/compute_distances) | NOT-STARTED | open prereq blocker **#941**. sklearn `__init__` (`_agglomerative.py:1296-1319`) takes 9 params with `n_clusters=2` default. `FeatureAgglomeration<F>` REQUIRES `n_clusters` (`fn new`, no default) and has only `linkage`/`pooling_func`. `distance_threshold` (cut by distance with `n_clusters=None`, `:1281`/`:1091-1092`) is materially absent. |
//! | REQ-7 (`pooling_func` as arbitrary callable) | NOT-STARTED | open prereq blocker **#941**. sklearn `_parameter_constraints["pooling_func"] = [callable]` (`_agglomerative.py:1291`) accepts any callable (default `np.mean`, `:1305`). ferrolearn offers only the closed `PoolingFunc::{Mean, Max}` enum (`fn with_pooling_func`). |
//! | REQ-8 (`inverse_transform`) | NOT-STARTED | open prereq blocker **#940**. sklearn `AgglomerationTransform.inverse_transform` broadcasts pooled values back via `X[..., np.unique(labels_, return_inverse=True)[1]]` (`_feature_agglomeration.py:66-92`). `FittedFeatureAgglomeration` impls only `Transform` — no `inverse_transform`. |
//! | REQ-9 (fitted attrs labels_/n_leaves_/n_connected_components_/children_/distances_) | NOT-STARTED | open prereq blocker **#942**. sklearn `_fit` sets `labels_`/`n_clusters_`/`n_leaves_`/`n_connected_components_`/`children_`/`distances_` (`_agglomerative.py:1083-1095`). `FittedFeatureAgglomeration` exposes `feature_labels_` (wrong name vs `labels_`, R-DEV-3), `n_clusters_`, `n_features_` — missing `n_leaves_`/`n_connected_components_`/`children_`/`distances_`. |
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
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> FeatureAgglomeration<F> {
    /// Create a new `FeatureAgglomeration` that reduces to `n_clusters` features.
    ///
    /// Defaults: `linkage = Ward`, `pooling_func = Mean`.
    #[must_use]
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            linkage: AgglomerativeLinkage::Ward,
            pooling_func: PoolingFunc::Mean,
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
    feature_labels_: Array1<usize>,
    /// Number of feature clusters.
    n_clusters_: usize,
    /// Number of original features.
    n_features_: usize,
    /// Pooling function to aggregate features within each cluster.
    pooling_func_: PoolingFunc,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> FittedFeatureAgglomeration<F> {
    /// Return the cluster label assigned to each feature.
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
        let agg = AgglomerativeClustering::<F>::new(self.n_clusters)
            .with_linkage(map_linkage(self.linkage));
        let fitted_agg = agg.fit(&x_t, &())?;

        Ok(FittedFeatureAgglomeration {
            feature_labels_: fitted_agg.labels_,
            n_clusters_: self.n_clusters,
            n_features_: n_features,
            pooling_func_: self.pooling_func,
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
