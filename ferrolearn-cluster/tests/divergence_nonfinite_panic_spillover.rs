//! Divergence pins (R-CODE-2 RELEASE-BLOCKER): cluster *spillover* estimators
//! that PANIC on non-finite (NaN/Inf) input, whereas scikit-learn 1.5.2 rejects
//! cleanly with `ValueError` via `check_array(force_all_finite=True)`.
//!
//! These are the same class of defect as the #2282 Agglomerative/GMM panics, in
//! estimators the #2282/#2283 fix did NOT cover. They lack any `is_finite`
//! up-front check (`grep -n is_finite src/{affinity_propagation,bisecting_kmeans}.rs`
//! → no matches), so a NaN/Inf flows into k-means++ centroid sampling and the
//! `rng.random_range(0.0..total)` (or `argmin`) aborts the process.
//!
//! sklearn rejects via `self._validate_data(...)` (`force_all_finite=True`
//! default, `sklearn/utils/validation.py:727`, `:1164`; raises at
//! `validation.py:147-154`):
//!   - AffinityPropagation.fit: `sklearn/cluster/_affinity_propagation.py:510`
//!       `X = self._validate_data(X, accept_sparse="csr")`
//!   - BisectingKMeans.fit:     `sklearn/cluster/_bisect_k_means.py:380`
//!       `X = self._validate_data(X, accept_sparse="csr", ...)`
//!
//! Live oracle (sklearn 1.5.2), from /tmp:
//!   AffinityPropagation(random_state=0).fit(<NaN X>)
//!     -> ValueError: Input X contains NaN.
//!   BisectingKMeans(n_clusters=2,random_state=0).fit(<NaN X>)
//!     -> ValueError: Input X contains NaN.
//!   BisectingKMeans(n_clusters=2,random_state=0).fit(<+Inf X>)
//!     -> ValueError: Input X contains infinity or a value too large for dtype('float64').
//!
//! ferrolearn actual (probed via catch_unwind):
//!   - AffinityPropagation::fit(NaN)  -> PANIC
//!   - BisectingKMeans::fit(NaN)      -> PANIC (kmeans++ `rng.random_range`,
//!     `bisecting_kmeans.rs:266-271`, NaN `total` → empty/NaN range)
//!   - BisectingKMeans::fit(+Inf)     -> PANIC
//!
//! A panic where sklearn raises a clean `ValueError` is an R-CODE-2 release-blocker.
//!
//! Tracking: #2285

use ferrolearn_cluster::{AffinityPropagation, BisectingKMeans};
use ferrolearn_core::Fit;
use ndarray::Array2;
use std::panic;

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

/// sklearn AffinityPropagation.fit(NaN) -> ValueError (clean). ferrolearn panics.
#[test]
#[ignore = "divergence: AffinityPropagation::fit panics on NaN; sklearn ValueError; tracking #2285"]
fn divergence_affinity_propagation_fit_nan_no_panic() {
    panic::set_hook(Box::new(|_| {}));
    let res = panic::catch_unwind(|| AffinityPropagation::<f64>::new().fit(&x_nan(), &()));
    assert!(
        res.is_ok(),
        "AffinityPropagation::fit panicked on NaN; sklearn raises ValueError cleanly"
    );
    assert!(
        res.unwrap().is_err(),
        "sklearn rejects NaN with ValueError; ferrolearn must return Err"
    );
}

/// sklearn BisectingKMeans.fit(NaN) -> ValueError (clean). ferrolearn panics
/// in kmeans++ `rng.random_range` (NaN `total`).
#[test]
#[ignore = "divergence: BisectingKMeans::fit panics on NaN; sklearn ValueError; tracking #2285"]
fn divergence_bisecting_kmeans_fit_nan_no_panic() {
    panic::set_hook(Box::new(|_| {}));
    let res = panic::catch_unwind(|| BisectingKMeans::new(2).with_random_state(0).fit(&x_nan(), &()));
    assert!(
        res.is_ok(),
        "BisectingKMeans::fit panicked on NaN; sklearn raises ValueError cleanly"
    );
    assert!(
        res.unwrap().is_err(),
        "sklearn rejects NaN with ValueError; ferrolearn must return Err"
    );
}

/// sklearn BisectingKMeans.fit(+Inf) -> ValueError (clean). ferrolearn panics.
#[test]
#[ignore = "divergence: BisectingKMeans::fit panics on Inf; sklearn ValueError; tracking #2285"]
fn divergence_bisecting_kmeans_fit_inf_no_panic() {
    panic::set_hook(Box::new(|_| {}));
    let res = panic::catch_unwind(|| BisectingKMeans::new(2).with_random_state(0).fit(&x_inf(), &()));
    assert!(
        res.is_ok(),
        "BisectingKMeans::fit panicked on Inf; sklearn raises ValueError cleanly"
    );
    assert!(
        res.unwrap().is_err(),
        "sklearn rejects Inf with ValueError; ferrolearn must return Err"
    );
}
