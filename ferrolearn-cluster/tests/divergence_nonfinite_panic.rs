//! Divergence pins (R-CODE-2 RELEASE-BLOCKER): `ferrolearn-cluster`
//! `AgglomerativeClustering::fit` and `GaussianMixture::fit` PANIC on NaN/Inf
//! input, whereas scikit-learn 1.5.2 rejects cleanly with `ValueError`.
//!
//! sklearn cluster/mixture estimators call `self._validate_data(...)` whose
//! `check_array(force_all_finite=True)` default (`sklearn/utils/validation.py:727`,
//! `:1164`) rejects BOTH NaN and Inf:
//!   - AgglomerativeClustering: `sklearn/cluster/_agglomerative.py:989`
//!         `X = self._validate_data(X, ensure_min_samples=2)`
//!   - GaussianMixture:         `sklearn/mixture/_base.py:212`
//!         `X = self._validate_data(X, dtype=[np.float64, np.float32], ensure_min_samples=2)`
//!
//! Live oracle (sklearn 1.5.2), from /tmp:
//!   python3 -c "import numpy as np; from sklearn.cluster import AgglomerativeClustering; \
//!     AgglomerativeClustering(n_clusters=2).fit(np.array([[1.,2.],[3.,np.nan],[5.,6.],[7.,8.]]))"
//!     -> ValueError: Input X contains NaN. AgglomerativeClustering does not accept missing values...
//!   (same class with np.inf -> ValueError: Input X contains infinity or a value too large...)
//!   python3 -c "... GaussianMixture(n_components=2).fit(<NaN X>)" -> ValueError: Input X contains NaN...
//!
//! ferrolearn actual (probed via catch_unwind): both PANIC.
//!   - AgglomerativeClustering: arithmetic underflow in `fn condensed_index`
//!     (`agglomerative.rs:339`) — NaN distances mis-route the nn-chain so a
//!     `usize` subtraction underflows.
//!   - GaussianMixture: `rng.random_range(0.0..total)` (`gmm.rs:605`) — NaN k-means++
//!     `total` makes the sampling range empty/NaN -> `random_range` panics.
//! A panic where sklearn raises a clean `ValueError` is an R-CODE-2 release-blocker.
//!
//! Tracking: #2282

use ferrolearn_cluster::{AgglomerativeClustering, GaussianMixture};
use ferrolearn_core::Fit;
use ndarray::Array2;
use std::panic;

fn x_nan() -> Array2<f64> {
    Array2::from_shape_vec(
        (6, 2),
        vec![1.0, 2.0, 3.0, f64::NAN, 5.0, 6.0, 7.0, 8.0, 1.1, 2.1, 3.1, 4.1],
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

/// sklearn raises ValueError (no panic); ferrolearn must NOT panic and SHOULD
/// return `Err`. Asserts no panic — FAILS today (Agglo panics in condensed_index).
#[test]
#[ignore = "divergence: AgglomerativeClustering::fit panics on NaN (sklearn raises ValueError); tracking #2282"]
fn divergence_agglomerative_fit_nan_no_panic() {
    panic::set_hook(Box::new(|_| {}));
    let res = panic::catch_unwind(|| AgglomerativeClustering::new(2).fit(&x_nan(), &()));
    // sklearn: clean ValueError. ferrolearn must not panic.
    assert!(
        res.is_ok(),
        "AgglomerativeClustering::fit panicked on NaN; sklearn raises ValueError cleanly"
    );
    // and, like sklearn, it should reject the bad input.
    assert!(
        res.unwrap().is_err(),
        "sklearn rejects NaN with ValueError; ferrolearn must return Err"
    );
}

#[test]
#[ignore = "divergence: AgglomerativeClustering::fit panics on +Inf (sklearn raises ValueError); tracking #2282"]
fn divergence_agglomerative_fit_inf_no_panic() {
    panic::set_hook(Box::new(|_| {}));
    let res = panic::catch_unwind(|| AgglomerativeClustering::new(2).fit(&x_inf(), &()));
    assert!(
        res.is_ok(),
        "AgglomerativeClustering::fit panicked on Inf; sklearn raises ValueError cleanly"
    );
    assert!(
        res.unwrap().is_err(),
        "sklearn rejects Inf with ValueError; ferrolearn must return Err"
    );
}

#[test]
#[ignore = "divergence: GaussianMixture::fit panics on NaN (sklearn raises ValueError); tracking #2282"]
fn divergence_gmm_fit_nan_no_panic() {
    panic::set_hook(Box::new(|_| {}));
    let res =
        panic::catch_unwind(|| GaussianMixture::<f64>::new(2).with_max_iter(10).fit(&x_nan(), &()));
    assert!(
        res.is_ok(),
        "GaussianMixture::fit panicked on NaN; sklearn raises ValueError cleanly"
    );
    assert!(
        res.unwrap().is_err(),
        "sklearn rejects NaN with ValueError; ferrolearn must return Err"
    );
}

#[test]
#[ignore = "divergence: GaussianMixture::fit panics on +Inf (sklearn raises ValueError); tracking #2282"]
fn divergence_gmm_fit_inf_no_panic() {
    panic::set_hook(Box::new(|_| {}));
    let res =
        panic::catch_unwind(|| GaussianMixture::<f64>::new(2).with_max_iter(10).fit(&x_inf(), &()));
    assert!(
        res.is_ok(),
        "GaussianMixture::fit panicked on Inf; sklearn raises ValueError cleanly"
    );
    assert!(
        res.unwrap().is_err(),
        "sklearn rejects Inf with ValueError; ferrolearn must return Err"
    );
}
