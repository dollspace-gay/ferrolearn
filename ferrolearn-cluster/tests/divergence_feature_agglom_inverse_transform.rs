//! Divergence pin for `FittedFeatureAgglomeration::inverse_transform`
//! (`ferrolearn-cluster/src/feature_agglomeration.rs:494-511`) vs the LIVE
//! scikit-learn 1.5.2 oracle (`sklearn.cluster.FeatureAgglomeration.inverse_transform`,
//! `sklearn/cluster/_feature_agglomeration.py:66-92`).
//!
//! sklearn's `inverse_transform` is purely `X[..., inverse]` (numpy fancy
//! indexing on the last axis, `_feature_agglomeration.py:92`) and performs NO
//! check that `xred` has exactly `n_clusters` columns. When `xred` has MORE
//! columns than `n_clusters`, numpy indexes only the columns named by
//! `inverse` (= `labels_`, values `0..n_clusters-1`) and silently ignores the
//! trailing extra column(s) — it returns a value, it does not raise.
//!
//! ferrolearn (`feature_agglomeration.rs:495-501`) instead REJECTS any
//! `xred.ncols() != n_clusters_` with `FerroError::ShapeMismatch`, so for the
//! `ncols > n_clusters` case it returns `Err` where sklearn returns a matrix.
//!
//! NOTE: the VALUE broadcast itself (column j scattered to all features with
//! `labels_ == j`) matches sklearn bit-for-bit for arbitrary `xred`, including
//! the permuted `_hc_cut` numbering `[0,0,2,2,1,1]` — that part is CLEAN
//! (verified separately). The divergence below is purely the over-wide
//! validation contract.
//!
//! Every expected value is a LIVE `sklearn` 1.5.2 oracle value (computed via
//! `python3 -c "..."`, quoted below) — NEVER copied from ferrolearn
//! (goal.md R-CHAR-3).

use ferrolearn_cluster::{AgglomerativeLinkage, FeatureAgglomeration};
use ferrolearn_core::traits::Fit;
use ndarray::Array2;

/// The 5x6 fixture with `_hc_cut` numbering `labels_ = [0,0,2,2,1,1]` (ward, k=3).
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

/// Divergence: `FittedFeatureAgglomeration::inverse_transform`
/// (`feature_agglomeration.rs:495-501`) rejects `xred` whose column count
/// EXCEEDS `n_clusters`, but
/// `sklearn/cluster/_feature_agglomeration.py:92` (`return X[..., inverse]`)
/// performs no such check: numpy fancy-indexing reads only columns
/// `inverse == labels_` (values `0..n_clusters-1`) and ignores the extra
/// trailing column, returning a value.
///
/// Input: `n_clusters=3`, `xred` of shape `(2, 4)` = `[[10,20,30,99],
/// [40,50,60,88]]` (one extra column 99/88).
///
/// Live sklearn 1.5.2 oracle (run from /tmp):
///   m = FeatureAgglomeration(n_clusters=3, linkage='ward').fit(X)
///   # m.labels_ == [0,0,2,2,1,1]
///   xred = np.array([[10.,20.,30.,99.],[40.,50.,60.,88.]])
///   m.inverse_transform(xred)
///   -> shape (2, 6); [[10,10,30,30,20,20],[40,40,60,60,50,50]]
///
/// sklearn returns the `(2,6)` matrix `[[10,10,30,30,20,20],[40,40,60,60,50,50]]`;
/// ferrolearn returns `Err(FerroError::ShapeMismatch)`.
///
/// Tracking: #2187
#[test]
fn divergence_inverse_transform_extra_cols_ignored() {
    let x = make_correlated_features();
    let fitted = FeatureAgglomeration::<f64>::new(3)
        .with_linkage(AgglomerativeLinkage::Ward)
        .fit(&x, &())
        .unwrap();

    // labels_ == [0,0,2,2,1,1] (matches sklearn oracle); the extra 4th column
    // (99/88) is never referenced by `inverse == labels_` and is dropped.
    let xred = Array2::from_shape_vec((2, 4), vec![10.0, 20.0, 30.0, 99.0, 40.0, 50.0, 60.0, 88.0])
        .unwrap();

    // sklearn live-oracle expected: (2,6), trailing column ignored.
    let sk_expected: [[f64; 6]; 2] = [
        [10.0, 10.0, 30.0, 30.0, 20.0, 20.0],
        [40.0, 40.0, 60.0, 60.0, 50.0, 50.0],
    ];

    let out = fitted.inverse_transform(&xred).expect(
        "sklearn returns a (2,6) matrix here; ferrolearn must not reject xred.ncols()>n_clusters",
    );

    assert_eq!(
        out.dim(),
        (2, 6),
        "sklearn inverse_transform(xred (2,4)) returns shape (2,6); got {:?}",
        out.dim()
    );
    for (i, row) in sk_expected.iter().enumerate() {
        for (f, &sk) in row.iter().enumerate() {
            assert!(
                (out[[i, f]] - sk).abs() < 1e-12,
                "inverse_transform[{i},{f}] = {} != sklearn {sk}",
                out[[i, f]]
            );
        }
    }
}
