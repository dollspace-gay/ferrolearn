//! ACToR critic re-audit (#2324): the `fit(train)` then `transform(test)` SPLIT
//! path for `ferrolearn-preprocess`'s `KNNImputer`
//! (`ferrolearn-preprocess/src/knn_imputer.rs`) vs scikit-learn 1.5.2
//! `KNNImputer.transform` (`sklearn/impute/_knn.py:246-377`).
//!
//! The existing `divergence_knn_imputer.rs` audit exercises only `fit_transform`
//! (fit data == transform data). sklearn's documented contract is that
//! `transform` imputes NEW rows using the FIT data as the donor pool
//! (`_knn.py:319` `non_missing_fix_X`; `_knn.py:353` `self._fit_X[...]`). This
//! file pins that split path with LIVE sklearn 1.5.2 oracle values (run from
//! `/tmp`, R-CHAR-3; NEVER copied from ferrolearn). Oracle version confirmed
//! `python3 -c "import sklearn; print(sklearn.__version__)"` -> `1.5.2`.

use approx::assert_abs_diff_eq;
use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_preprocess::knn_imputer::{KNNImputer, KNNWeights};
use ndarray::array;

/// Guard (#7 fit-vs-transform, UNIFORM): fit on a 4-row donor pool, transform two
/// disjoint test rows. Receiver `[2.5, nan]` draws its col-1 donors from the FIT
/// data: nearest two by col-0 distance are train rows 1 (2.0) and 2 (3.0), so the
/// uniform mean is `mean(20, 30) = 25.0`. The no-missing test row `[100, 5]`
/// passes through unchanged.
///
/// Live oracle:
/// `KNNImputer(n_neighbors=2).fit([[1,10],[2,20],[3,30],[4,40]]).transform(
///   [[2.5, nan],[100, 5]])` -> `[[2.5, 25.0], [100.0, 5.0]]`.
#[test]
fn guard_knn_fit_transform_split_uniform() {
    let x_train = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]];
    let x_test = array![[2.5, f64::NAN], [100.0, 5.0]];
    let fitted = KNNImputer::<f64>::new(2, KNNWeights::Uniform)
        .fit(&x_train, &())
        .unwrap();
    let out = fitted.transform(&x_test).unwrap();
    assert_abs_diff_eq!(out[[0, 1]], 25.0, epsilon = 1e-9);
    assert_abs_diff_eq!(out[[1, 0]], 100.0, epsilon = 1e-9);
    assert_abs_diff_eq!(out[[1, 1]], 5.0, epsilon = 1e-9);
}

/// Guard (#7 fit-vs-transform, DISTANCE): same fit pool, distance weights.
///
/// Live oracle:
/// `KNNImputer(n_neighbors=2, weights='distance').fit([[1,10],[2,20],[3,30],[4,40]])
///   .transform([[2.5, nan],[10.0, nan]])`
/// -> `[[2.5, 24.999999999999996], [10.0, 35.38461538461539]]`.
#[test]
fn guard_knn_fit_transform_split_distance() {
    let x_train = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]];
    let x_test = array![[2.5, f64::NAN], [10.0, f64::NAN]];
    let fitted = KNNImputer::<f64>::new(2, KNNWeights::Distance)
        .fit(&x_train, &())
        .unwrap();
    let out = fitted.transform(&x_test).unwrap();
    assert_abs_diff_eq!(out[[0, 1]], 24.999_999_999_999_996, epsilon = 1e-9);
    assert_abs_diff_eq!(out[[1, 1]], 35.384_615_384_615_39, epsilon = 1e-9);
}
