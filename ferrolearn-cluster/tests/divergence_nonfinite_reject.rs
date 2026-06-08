//! Divergence pins: `ferrolearn-cluster` KMeans / DBSCAN / MeanShift `fit` and
//! KMeans / GaussianMixture `predict` SILENTLY ACCEPT NaN/Inf and return an
//! Ok-garbage clustering, whereas scikit-learn 1.5.2 rejects with `ValueError`.
//!
//! sklearn rejects NaN AND Inf via `check_array(force_all_finite=True)`
//! (`sklearn/utils/validation.py:727`, `:1164`), reached from:
//!   - KMeans.fit:       `sklearn/cluster/_kmeans.py:1464`  `X = self._validate_data(...)`
//!   - KMeans.predict:   `sklearn/cluster/_kmeans.py:1091` -> `_check_test_data` (`:950`) -> `check_array`
//!   - DBSCAN.fit:       `sklearn/cluster/_dbscan.py:395`   `X = self._validate_data(X, accept_sparse="csr")`
//!   - MeanShift.fit:    `sklearn/cluster/_mean_shift.py:485` `X = self._validate_data(X)`
//!   - GaussianMixture.predict: `sklearn/mixture/_base.py:384` `X = self._validate_data(X, reset=False)`
//!
//! Live oracle (sklearn 1.5.2), from /tmp — every call below raises ValueError:
//!   KMeans(n_clusters=2,n_init=2,random_state=0).fit(<NaN X>) -> ValueError: Input X contains NaN...
//!   KMeans(...).fit(<finite X>).predict(np.array([[1.,np.nan]])) -> ValueError: Input X contains NaN...
//!   DBSCAN(eps=0.5,min_samples=2).fit(<NaN X>) -> ValueError: Input X contains NaN...
//!   MeanShift(bandwidth=2.0).fit(<NaN X>) -> ValueError: Input X contains NaN...
//!   GaussianMixture(n_components=2,random_state=0).fit(<finite X>).predict(<NaN X>) -> ValueError...
//!   (each also raises "Input X contains infinity..." for the +Inf variant.)
//!
//! ferrolearn actual (probed): KMeans/DBSCAN/MeanShift fit -> Ok(garbage);
//! KMeans/GMM predict -> Ok(garbage). None reject; none have any finite-check.
//! Per R-DEV-1 (NaN/Inf handling matches sklearn), these must return Err.
//!
//! Tracking: #2283

use ferrolearn_cluster::{DBSCAN, GaussianMixture, KMeans, MeanShift};
use ferrolearn_core::{Fit, Predict};
use ndarray::Array2;

fn x_nan() -> Array2<f64> {
    Array2::from_shape_vec(
        (6, 2),
        vec![
            1.0,
            2.0,
            3.0,
            f64::NAN,
            5.0,
            6.0,
            7.0,
            8.0,
            1.1,
            2.1,
            3.1,
            4.1,
        ],
    )
    .unwrap()
}
fn x_inf() -> Array2<f64> {
    Array2::from_shape_vec(
        (6, 2),
        vec![
            1.0,
            2.0,
            3.0,
            f64::INFINITY,
            5.0,
            6.0,
            7.0,
            8.0,
            1.1,
            2.1,
            3.1,
            4.1,
        ],
    )
    .unwrap()
}
fn x_finite() -> Array2<f64> {
    Array2::from_shape_vec(
        (6, 2),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.1, 2.1, 3.1, 4.1],
    )
    .unwrap()
}

#[test]
fn divergence_kmeans_fit_rejects_nan() {
    let r = KMeans::new(2)
        .with_n_init(2)
        .with_random_state(0)
        .fit(&x_nan(), &());
    assert!(
        r.is_err(),
        "sklearn KMeans.fit raises ValueError on NaN; ferrolearn must Err"
    );
}

#[test]
fn divergence_kmeans_fit_rejects_inf() {
    let r = KMeans::new(2)
        .with_n_init(2)
        .with_random_state(0)
        .fit(&x_inf(), &());
    assert!(
        r.is_err(),
        "sklearn KMeans.fit raises ValueError on Inf; ferrolearn must Err"
    );
}

#[test]
fn divergence_dbscan_fit_rejects_nan() {
    let r = DBSCAN::new(0.5).with_min_samples(2).fit(&x_nan(), &());
    assert!(
        r.is_err(),
        "sklearn DBSCAN.fit raises ValueError on NaN; ferrolearn must Err"
    );
}

#[test]
fn divergence_dbscan_fit_rejects_inf() {
    let r = DBSCAN::new(0.5).with_min_samples(2).fit(&x_inf(), &());
    assert!(
        r.is_err(),
        "sklearn DBSCAN.fit raises ValueError on Inf; ferrolearn must Err"
    );
}

#[test]
fn divergence_mean_shift_fit_rejects_nan() {
    let r = MeanShift::new().with_bandwidth(2.0).fit(&x_nan(), &());
    assert!(
        r.is_err(),
        "sklearn MeanShift.fit raises ValueError on NaN; ferrolearn must Err"
    );
}

#[test]
fn divergence_mean_shift_fit_rejects_inf() {
    let r = MeanShift::new().with_bandwidth(2.0).fit(&x_inf(), &());
    assert!(
        r.is_err(),
        "sklearn MeanShift.fit raises ValueError on Inf; ferrolearn must Err"
    );
}

#[test]
fn divergence_kmeans_predict_rejects_nan() {
    let fitted = KMeans::new(2)
        .with_n_init(2)
        .with_random_state(0)
        .fit(&x_finite(), &())
        .expect("finite fit ok");
    let q = Array2::from_shape_vec((1, 2), vec![1.0, f64::NAN]).unwrap();
    let r = fitted.predict(&q);
    assert!(
        r.is_err(),
        "sklearn KMeans.predict raises ValueError on NaN; ferrolearn must Err"
    );
}

#[test]
fn divergence_gmm_predict_rejects_nan() {
    let fitted = GaussianMixture::<f64>::new(2)
        .with_max_iter(10)
        .fit(&x_finite(), &())
        .expect("finite fit ok");
    let q = Array2::from_shape_vec((1, 2), vec![1.0, f64::NAN]).unwrap();
    let r = fitted.predict(&q);
    assert!(
        r.is_err(),
        "sklearn GaussianMixture.predict raises ValueError on NaN; ferrolearn must Err"
    );
}
