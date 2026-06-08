//! Divergence pins: cluster *spillover* estimators that SILENTLY ACCEPT
//! non-finite (NaN/Inf) input and return Ok-garbage, whereas scikit-learn 1.5.2
//! rejects with `ValueError` via `check_array(force_all_finite=True)`.
//!
//! These estimators were NOT covered by the #2282/#2283 finite-check fix and
//! have no `is_finite` guard (`grep -n is_finite src/{spectral,affinity_propagation,
//! mini_batch_kmeans,label_propagation,label_spreading}.rs` → no matches).
//!
//! sklearn rejects via `self._validate_data(...)` (`force_all_finite=True`
//! default, `sklearn/utils/validation.py:727`, `:1164`; raises at
//! `validation.py:147-154`):
//!   - SpectralClustering.fit: `sklearn/cluster/_spectral.py:691`
//!     `X = self._validate_data(X, accept_sparse=["csr","csc","coo"], dtype=np.float64, ensure_min_samples=2)`
//!   - AffinityPropagation.fit: `sklearn/cluster/_affinity_propagation.py:510`
//!     `X = self._validate_data(X, accept_sparse="csr")`
//!   - MiniBatchKMeans.fit: `sklearn/cluster/_kmeans.py:2073`
//!     `X = self._validate_data(X, accept_sparse="csr", dtype=[np.float64, np.float32], ...)`
//!   - LabelPropagation.fit / LabelSpreading.fit: `sklearn/semi_supervised/_label_propagation.py:258`
//!     `X, y = self._validate_data(X, y, accept_sparse=["csr","csc"], reset=True)`
//!
//! Live oracle (sklearn 1.5.2), from /tmp — each raises:
//!   SpectralClustering(n_clusters=2,random_state=0).fit(<+Inf X>)
//!     -> ValueError: Input X contains infinity or a value too large for dtype('float64').
//!   AffinityPropagation(random_state=0).fit(<+Inf X>)
//!     -> ValueError: Input X contains infinity or a value too large for dtype('float64').
//!   MiniBatchKMeans(n_clusters=2,random_state=0,n_init=2).fit(<NaN X>)
//!     -> ValueError: Input X contains NaN.
//!   MiniBatchKMeans(...).fit(<+Inf X>)
//!     -> ValueError: Input X contains infinity ...
//!   LabelPropagation().fit(<NaN X>, y)  -> ValueError: Input X contains NaN.
//!   LabelSpreading().fit(<NaN X>, y)    -> ValueError: Input X contains NaN.
//!
//! ferrolearn actual (probed): each returns Ok(garbage); none reject.
//! Per R-DEV-1 (NaN/Inf handling matches sklearn), these must return Err.
//!
//! NOTE on scope (R-HONEST): SpectralClustering ALREADY rejects NaN (Err) but
//! NOT +Inf; AffinityPropagation NaN PANICS (pinned separately in
//! `divergence_nonfinite_panic_spillover.rs`) but +Inf silently accepts.
//! Birch and FeatureAgglomeration already reject BOTH NaN and Inf (inherited
//! from the AgglomerativeClustering fix) — NOT pinned. HDBSCAN is NOT pinned:
//! sklearn HDBSCAN uses `force_all_finite=False` (`_hdbscan/hdbscan.py:737,:782`)
//! and itself ACCEPTS NaN/Inf, so ferrolearn accepting it is parity, not divergence.
//!
//! Tracking: #2286

use ferrolearn_cluster::{
    AffinityPropagation, LabelPropagation, LabelSpreading, MiniBatchKMeans, SpectralClustering,
};
use ferrolearn_core::Fit;
use ndarray::{Array1, Array2};

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
fn y_semi() -> Array1<isize> {
    Array1::from(vec![0isize, 1, -1, -1, 0, 1])
}

#[test]
fn divergence_spectral_fit_rejects_inf() {
    let r = SpectralClustering::new(2)
        .with_random_state(0)
        .fit(&x_inf(), &());
    assert!(
        r.is_err(),
        "sklearn SpectralClustering.fit raises ValueError on Inf; ferrolearn must Err"
    );
}

#[test]
fn divergence_affinity_propagation_fit_rejects_inf() {
    let r = AffinityPropagation::<f64>::new().fit(&x_inf(), &());
    assert!(
        r.is_err(),
        "sklearn AffinityPropagation.fit raises ValueError on Inf; ferrolearn must Err"
    );
}

#[test]
fn divergence_mini_batch_kmeans_fit_rejects_nan() {
    let r = MiniBatchKMeans::new(2)
        .with_random_state(0)
        .with_n_init(2)
        .fit(&x_nan(), &());
    assert!(
        r.is_err(),
        "sklearn MiniBatchKMeans.fit raises ValueError on NaN; ferrolearn must Err"
    );
}

#[test]
fn divergence_mini_batch_kmeans_fit_rejects_inf() {
    let r = MiniBatchKMeans::new(2)
        .with_random_state(0)
        .with_n_init(2)
        .fit(&x_inf(), &());
    assert!(
        r.is_err(),
        "sklearn MiniBatchKMeans.fit raises ValueError on Inf; ferrolearn must Err"
    );
}

#[test]
fn divergence_label_propagation_fit_rejects_nan() {
    let r = LabelPropagation::<f64>::new().fit(&x_nan(), &y_semi());
    assert!(
        r.is_err(),
        "sklearn LabelPropagation.fit raises ValueError on NaN; ferrolearn must Err"
    );
}

#[test]
fn divergence_label_propagation_fit_rejects_inf() {
    let r = LabelPropagation::<f64>::new().fit(&x_inf(), &y_semi());
    assert!(
        r.is_err(),
        "sklearn LabelPropagation.fit raises ValueError on Inf; ferrolearn must Err"
    );
}

#[test]
fn divergence_label_spreading_fit_rejects_nan() {
    let r = LabelSpreading::<f64>::new().fit(&x_nan(), &y_semi());
    assert!(
        r.is_err(),
        "sklearn LabelSpreading.fit raises ValueError on NaN; ferrolearn must Err"
    );
}

#[test]
fn divergence_label_spreading_fit_rejects_inf() {
    let r = LabelSpreading::<f64>::new().fit(&x_inf(), &y_semi());
    assert!(
        r.is_err(),
        "sklearn LabelSpreading.fit raises ValueError on Inf; ferrolearn must Err"
    );
}
