//! Divergence pins + value-contract guards for `FeatureAgglomeration` /
//! `FittedFeatureAgglomeration` (`ferrolearn-cluster/src/feature_agglomeration.rs`)
//! against the LIVE scikit-learn 1.5.2 oracle (`from sklearn.cluster import
//! FeatureAgglomeration`, mirroring `sklearn/cluster/_agglomerative.py` +
//! `sklearn/cluster/_feature_agglomeration.py`).
//!
//! Every expected value below is a LIVE `sklearn` 1.5.2 oracle value (computed
//! via `python3 -c "..."` run from `/tmp`, quoted above each block) — NEVER
//! copied from the ferrolearn side (goal.md R-CHAR-3).
//!
//! Design doc: `.design/cluster/feature_agglomeration.md` (commit 2f7ebc37).
//!
//! ## Test taxonomy
//!
//! RED pin (FAILS now, LIVE `#[test]`, NO `#[ignore]` — the one clean
//! feature_agglomeration.rs-LOCAL fixable divergence):
//! - `divergence_feature_agglom_min_features_two` (RED, #944) — sklearn
//!   `FeatureAgglomeration.fit` calls `_validate_data(X, ensure_min_features=2)`
//!   (`_agglomerative.py:1338`) and RAISES `ValueError` on a 1-feature `X`.
//!   ferrolearn's `fn fit` only checks `n_features >= n_clusters` (1 >= 1 passes)
//!   and `n_samples >= 1`, so it ACCEPTS a 1-column `X` — over-acceptance.
//!   Minimally fixable: add an `n_features < 2` reject in `fn fit`.
//!
//! GREEN guards (PASS now — protect the parts already correct):
//! - `green_feature_agglom_transform_shape` (GREEN) — `transform(X)` shape is
//!   `(n_samples, n_clusters)`. SHAPE only (column VALUES/order diverge — see the
//!   documented #938 note — and are NOT asserted here).
//! - `green_feature_agglom_mean_pooling_as_set` (GREEN) — the mean-pooling
//!   ARITHMETIC matches sklearn even though column ORDER is permuted: for each
//!   row, ferrolearn's set of pooled column values (sorted) equals sklearn's
//!   sorted row. Guards the pooling arithmetic WITHOUT overclaiming column-order
//!   parity (#938).
//! - `green_feature_agglom_n_clusters_zero_rejected` (GREEN) — both reject
//!   `n_clusters=0` (sklearn `Interval(Integral, 1, None)` `_agglomerative.py:1281`
//!   → `InvalidParameterError`; ferrolearn `FerroError::InvalidParameter`).
//! - `green_feature_agglom_too_many_clusters_rejected` (GREEN) — both reject
//!   `n_clusters > n_features` (sklearn `ValueError` "Cannot extract more clusters
//!   than samples", from `_hc_cut`; ferrolearn `FerroError::InvalidParameter`).
//!
//! ## Documented NOT-STARTED / cross-unit (NO forced test):
//!
//! label-numbering convention + transform column-order parity (#938): ferrolearn
//! groups features into exactly the SAME partition as sklearn (`{0,1}`, `{2,3}`,
//! `{4,5}` on the fixture below — Probe 1 of the design doc), but assigns a
//! PERMUTED integer label index: sklearn `_hc_cut` (`_agglomerative.py:1099`) +
//! `np.searchsorted(np.unique(labels), labels)` (`:1105`) → `labels_ =
//! [0,0,2,2,1,1]`; ferrolearn relabels by `active`-slot order in
//! `agglomerative.rs::agglomerate` → `[0,0,1,1,2,2]`. sklearn's `transform` orders
//! output columns by `np.unique(self.labels_)` (`_feature_agglomeration.py:62`),
//! so the column ORDER diverges (sklearn mean row 0 = `[1.05, 9.05, 5.05]` vs
//! ferrolearn `[1.05, 5.05, 9.05]`). The root cause is OWNED by the
//! AgglomerativeClustering unit (`agglomerative.rs`), NOT
//! `feature_agglomeration.rs` — so a `labels_`/column-order equality test is NOT
//! pinned here (it would require the agglomerative.rs fix). Documented only; the
//! `green_feature_agglom_mean_pooling_as_set` guard below confirms the pooling
//! arithmetic is correct as an unordered set.
//!
//! inverse_transform (#940): sklearn `AgglomerationTransform.inverse_transform`
//! (`_feature_agglomeration.py:66-92`) broadcasts pooled values back to original
//! feature positions; `FittedFeatureAgglomeration` impls only `Transform`. Missing
//! surface.
//!
//! missing params + n_clusters=2 default + callable pooling_func (#941): sklearn
//! `__init__` (`_agglomerative.py:1296-1319`) takes 9 params with `n_clusters=2`
//! default and `pooling_func=np.mean` (any callable, `_parameter_constraints
//! ["pooling_func"] = [callable]` `:1291`); ferrolearn `fn new(n_clusters)`
//! REQUIRES n_clusters (no default) and has only `linkage`/`pooling_func`
//! (closed `PoolingFunc::{Mean, Max}` enum), missing
//! `metric`/`connectivity`/`distance_threshold`/`compute_full_tree`/`memory`/
//! `compute_distances`. Missing surface.
//!
//! fitted attrs children_/distances_/n_leaves_/n_connected_components_ +
//! labels_ name (#942): sklearn `_fit` (`_agglomerative.py:1083-1095`) sets
//! `labels_`/`n_clusters_`/`n_leaves_`/`n_connected_components_`/`children_`/
//! `distances_`; `FittedFeatureAgglomeration` exposes `feature_labels_` (wrong
//! name vs `labels_`), `n_clusters_`, `n_features_` only. Missing surface.
//!
//! PyO3 + ferray (#943): `grep -rln FeatureAgglomeration ferrolearn-python/` is
//! EMPTY — no `_RsFeatureAgglomeration`, so `import ferrolearn` cannot reach it;
//! `feature_agglomeration.rs` imports `ndarray` + `num-traits`, not `ferray-core`.
//! Missing binding / substrate.

use ferrolearn_cluster::{FeatureAgglomeration, PoolingFunc};
use ferrolearn_core::traits::{Fit, Transform};
use ndarray::Array2;

/// The `make_correlated_features` 5x6 fixture from the in-tree
/// `feature_agglomeration.rs` tests: 6 features paired `(0,1)`/`(2,3)`/`(4,5)`.
/// This is the same matrix the design doc's live oracle probes use.
fn make_correlated_features() -> Array2<f64> {
    Array2::from_shape_vec(
        (5, 6),
        vec![
            1.0, 1.1, 5.0, 5.1, 9.0, 9.1, 2.0, 2.1, 6.0, 6.1, 8.0, 8.1, 3.0, 3.1, 7.0, 7.1, 7.0,
            7.1, 4.0, 4.1, 8.0, 8.1, 6.0, 6.1, 5.0, 5.1, 9.0, 9.1, 5.0, 5.1,
        ],
    )
    .unwrap()
}

// ===========================================================================
// RED #944 — ensure_min_features=2: sklearn REJECTS a single-feature X.
//
// sklearn `FeatureAgglomeration.fit` calls
// `X = self._validate_data(X, ensure_min_features=2)`
// (`_agglomerative.py:1338`), so a 1-column X is rejected with a `ValueError`
// BEFORE any clustering happens. ferrolearn `fn fit` validates only
// `n_clusters > 0`, `n_features >= n_clusters` (1 >= 1 → passes), and
// `n_samples >= 1` — there is NO min-features guard, so a 1-column X with
// n_clusters=1 is ACCEPTED. Over-acceptance.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "import numpy as np; from sklearn.cluster import FeatureAgglomeration
//     FeatureAgglomeration(n_clusters=1).fit(np.array([[1.],[2.],[3.],[4.]]))"
//   ->  ValueError: Found array with 1 feature(s) (shape=(4, 1)) while a
//       minimum of 2 is required by FeatureAgglomeration.
//
// We pin the OBSERVABLE: ferrolearn `fit` on a (4,1) X must return `Err`
// (sklearn raises). ferrolearn currently returns `Ok`: FAILS now. Minimally
// fixable in `feature_agglomeration.rs` `fn fit` — add an `n_features < 2`
// reject (the `ensure_min_features=2` guard).
// ===========================================================================

/// Divergence: ferrolearn's `FeatureAgglomeration::fit` ACCEPTS a single-feature
/// `X`, whereas `sklearn/cluster/_agglomerative.py:1338`
/// (`X = self._validate_data(X, ensure_min_features=2)`) REJECTS it with a
/// `ValueError` ("Found array with 1 feature(s) ... while a minimum of 2 is
/// required by FeatureAgglomeration."). sklearn `FeatureAgglomeration(n_clusters=1)
/// .fit(X_1col)` raises; ferrolearn `fit` returns `Ok` because `fn fit` only
/// checks `n_features >= n_clusters` (1 >= 1) — no min-features guard.
/// Tracking: #944
#[test]
fn divergence_feature_agglom_min_features_two() {
    // (4, 1): a single feature — sklearn raises ValueError (oracle above).
    let x_1col = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

    let result = FeatureAgglomeration::<f64>::new(1).fit(&x_1col, &());

    assert!(
        result.is_err(),
        "fit on a (4,1) single-feature X should error (sklearn: \
         _validate_data(ensure_min_features=2) raises ValueError 'a minimum of 2 \
         is required by FeatureAgglomeration'), got Ok (#944)"
    );
}

// ===========================================================================
// GREEN — transform output shape is (n_samples, n_clusters).
//
// sklearn `FeatureAgglomeration(n_clusters=3).fit(X).transform(X)` returns an
// array of shape `(n_samples, n_clusters)` (one pooled column per cluster).
// ferrolearn's `fn transform` allocates `(n_samples, self.n_clusters_)`. SHAPE
// matches; column VALUES/order DIVERGE (#938) and are NOT asserted here.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "import numpy as np; from sklearn.cluster import FeatureAgglomeration
//     X=np.array([[1.,1.1,5.,5.1,9.,9.1],[2.,2.1,6.,6.1,8.,8.1],
//                 [3.,3.1,7.,7.1,7.,7.1],[4.,4.1,8.,8.1,6.,6.1],
//                 [5.,5.1,9.,9.1,5.,5.1]])
//     print(FeatureAgglomeration(n_clusters=3, pooling_func=np.mean).fit(X)
//           .transform(X).shape)"
//   ->  (5, 3)
// ===========================================================================

/// Guard: ferrolearn `FeatureAgglomeration::new(3).fit(X).transform(X)` has shape
/// `(5, 3)`, matching sklearn `transform(X).shape == (5, 3)` (n_samples=5,
/// n_clusters=3). Shape contract only — column values/order diverge (#938).
#[test]
fn green_feature_agglom_transform_shape() {
    let x = make_correlated_features();
    let fitted = FeatureAgglomeration::<f64>::new(3).fit(&x, &()).unwrap();
    let reduced = fitted.transform(&x).unwrap();

    // sklearn oracle: transform(X).shape == (5, 3).
    assert_eq!(
        reduced.dim(),
        (5, 3),
        "transform shape should be (n_samples=5, n_clusters=3) (sklearn oracle: \
         (5, 3))"
    );
}

// ===========================================================================
// GREEN — mean-pooling ARITHMETIC matches sklearn as an unordered column SET.
//
// The pooling arithmetic (mean of each feature group) is correct in ferrolearn;
// only the column ORDER diverges (#938, cross-unit). So we compare each row's
// pooled values as a SORTED set: sklearn's sorted row must equal ferrolearn's
// sorted row, element-wise. This guards the mean-pooling math WITHOUT
// overclaiming column-order parity.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "import numpy as np; from sklearn.cluster import FeatureAgglomeration
//     X=np.array([[1.,1.1,5.,5.1,9.,9.1],[2.,2.1,6.,6.1,8.,8.1],
//                 [3.,3.1,7.,7.1,7.,7.1],[4.,4.1,8.,8.1,6.,6.1],
//                 [5.,5.1,9.,9.1,5.,5.1]])
//     t=FeatureAgglomeration(n_clusters=3, pooling_func=np.mean).fit(X).transform(X)
//     print([sorted([round(float(v),10) for v in r]) for r in t])"
//   ->  [[1.05, 5.05, 9.05], [2.05, 6.05, 8.05], [3.05, 7.05, 7.05],
//        [4.05, 6.05, 8.05], [5.05, 5.05, 9.05]]
//   (NOTE: each row's pooled values SORTED; the raw column order in sklearn is
//    [1.05, 9.05, 5.05], ... but as an unordered set the rows agree with
//    ferrolearn.)
// ===========================================================================

/// Guard: ferrolearn `transform` (mean pooling) produces, per row, the same
/// multiset of pooled values as sklearn `FeatureAgglomeration(n_clusters=3,
/// pooling_func=np.mean).transform(X)` — compared as SORTED rows. This guards the
/// mean-pooling ARITHMETIC without asserting column ORDER (which diverges, #938,
/// owned by the AgglomerativeClustering unit). sklearn sorted rows (oracle above):
/// `[[1.05,5.05,9.05],[2.05,6.05,8.05],[3.05,7.05,7.05],[4.05,6.05,8.05],
/// [5.05,5.05,9.05]]`.
#[test]
fn green_feature_agglom_mean_pooling_as_set() {
    // sklearn oracle: each transform row, SORTED ascending.
    let sklearn_sorted_rows: [[f64; 3]; 5] = [
        [1.05, 5.05, 9.05],
        [2.05, 6.05, 8.05],
        [3.05, 7.05, 7.05],
        [4.05, 6.05, 8.05],
        [5.05, 5.05, 9.05],
    ];

    let x = make_correlated_features();
    let fitted = FeatureAgglomeration::<f64>::new(3)
        .with_pooling_func(PoolingFunc::Mean)
        .fit(&x, &())
        .unwrap();
    let reduced = fitted.transform(&x).unwrap();

    for (i, expected) in sklearn_sorted_rows.iter().enumerate() {
        let mut row: Vec<f64> = reduced.row(i).iter().copied().collect();
        row.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (j, &exp) in expected.iter().enumerate() {
            assert!(
                (row[j] - exp).abs() < 1e-9,
                "row {i} sorted-pooled value {j}: ferrolearn {} vs sklearn {exp} \
                 (mean-pooling arithmetic as an unordered set)",
                row[j]
            );
        }
    }
}

// ===========================================================================
// GREEN — n_clusters = 0 is REJECTED at fit (both sides reject).
//
// sklearn `FeatureAgglomeration._parameter_constraints["n_clusters"]` is
// `Interval(Integral, 1, None, closed="left")` (`_agglomerative.py:1281`) → 0 is
// OUTSIDE [1, inf) and raises `InvalidParameterError`. ferrolearn `fn fit`
// rejects `n_clusters == 0` with `FerroError::InvalidParameter`. Both error.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "import numpy as np; from sklearn.cluster import FeatureAgglomeration
//     FeatureAgglomeration(n_clusters=0).fit(np.random.rand(5,6))"
//   ->  InvalidParameterError  (n_clusters must be in [1, inf))
// ===========================================================================

/// Guard: ferrolearn `new(0).fit` returns `Err`, matching sklearn which raises
/// `InvalidParameterError` for `n_clusters=0` (outside `Interval(Integral, 1,
/// None, closed="left")`, `_agglomerative.py:1281`).
#[test]
fn green_feature_agglom_n_clusters_zero_rejected() {
    let x = make_correlated_features();
    let result = FeatureAgglomeration::<f64>::new(0).fit(&x, &());

    assert!(
        result.is_err(),
        "fit with n_clusters=0 should error (sklearn: InvalidParameterError, \
         n_clusters must be in [1, inf)), got Ok"
    );
}

// ===========================================================================
// GREEN — n_clusters > n_features is REJECTED at fit (both sides reject).
//
// sklearn rejects `n_clusters=10` on a 6-feature X via `_hc_cut` → `ValueError`
// ("Cannot extract more clusters than samples: 10 clusters were given for a tree
// with 6 leaves"). ferrolearn `fn fit` rejects `n_features < n_clusters` with
// `FerroError::InvalidParameter` ("n_clusters (10) exceeds n_features (6)").
// Both error.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "import numpy as np; from sklearn.cluster import FeatureAgglomeration
//     FeatureAgglomeration(n_clusters=10).fit(np.random.rand(5,6))"
//   ->  ValueError: Cannot extract more clusters than samples: 10 clusters were
//       given for a tree with 6 leaves.
// ===========================================================================

/// Guard: ferrolearn `new(10).fit(X_6col)` returns `Err` (n_clusters=10 >
/// n_features=6), matching sklearn which raises `ValueError` ("Cannot extract
/// more clusters than samples: 10 clusters were given for a tree with 6 leaves").
#[test]
fn green_feature_agglom_too_many_clusters_rejected() {
    // make_correlated_features has 6 columns; request 10 clusters.
    let x = make_correlated_features();
    let result = FeatureAgglomeration::<f64>::new(10).fit(&x, &());

    assert!(
        result.is_err(),
        "fit with n_clusters=10 > n_features=6 should error (sklearn: ValueError, \
         'Cannot extract more clusters than samples'), got Ok"
    );
}
