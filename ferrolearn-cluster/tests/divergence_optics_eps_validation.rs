//! Divergence pin: `ferrolearn_cluster::OPTICS` does not validate the `eps`
//! parameter against sklearn's constraint
//! `"eps": [Interval(Real, 0, None, closed="both"), None]`
//! (`sklearn/cluster/_optics.py:252`).
//!
//! sklearn validates `_parameter_constraints` via the `@_fit_context` /
//! `validate_params` wrapper BEFORE the fit body runs, so `OPTICS(eps=NaN)` and
//! `OPTICS(eps=-1.0)` raise `InvalidParameterError` for BOTH `cluster_method`
//! values (the constraint is on the estimator parameter, not gated on the
//! method). The recent validation fix (#2197) added explicit `is_nan()` guards
//! for `max_eps` and `xi`, but the `eps` parameter has NO NaN/negative guard:
//! the only `eps` check in `Fit::fit` is `eps > self.max_eps` in the Dbscan arm
//! (`optics.rs` `Fit::fit`), and `NaN > x` / `-1.0 > 5.0` are both `false`, so
//! the bad `eps` slips through. On the Xi path `eps` is never read at all.
//!
//! Live oracle (sklearn 1.5.2):
//! ```text
//! python3 -c "
//! import numpy as np
//! from sklearn.cluster import OPTICS
//! X = np.array([[0.,0.],[.1,.1],[0.,.1],[5.,5.],[5.1,5.],[5.,5.1],[10.,0.],[10.1,0.],[10.,.1]])
//! OPTICS(min_samples=2, cluster_method='dbscan', eps=float('nan'), max_eps=5.0).fit(X)"
//! # -> InvalidParameterError: The 'eps' parameter of OPTICS must be a float in
//! #    the range [0.0, inf] ...
//! ```
//! ferrolearn returns `Ok(..)` instead of `Err(..)`.

use ferrolearn_cluster::OPTICS;
use ferrolearn_cluster::optics::OpticsClusterMethod;
use ferrolearn_core::Fit;
use ndarray::Array2;

/// The 9-point three-blobs f64 fixture used across the OPTICS divergence suite.
fn three_blobs() -> Array2<f64> {
    Array2::from_shape_vec(
        (9, 2),
        vec![
            0.0, 0.0, 0.1, 0.1, 0.0, 0.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1, 10.0, 0.0, 10.1, 0.0,
            10.0, 0.1,
        ],
    )
    .unwrap()
}

/// Divergence: ferrolearn's `OPTICS::fit` accepts `eps=NaN` on the Dbscan path,
/// whereas sklearn rejects it via `_parameter_constraints`
/// (`sklearn/cluster/_optics.py:252`, `"eps": [Interval(Real, 0, None,
/// closed="both"), None]` — the `Interval` rejects NaN).
///
/// Oracle: `OPTICS(min_samples=2, cluster_method='dbscan', eps=nan,
/// max_eps=5.0).fit(X)` raises `InvalidParameterError`. ferrolearn returns `Ok`.
///
/// Tracking: #2198
#[test]
#[ignore = "divergence: OPTICS accepts eps=NaN / eps<0; tracking #2198"]
fn divergence_eps_nan_dbscan_rejected() {
    let x = three_blobs();
    let result = OPTICS::<f64>::new(2)
        .with_max_eps(5.0)
        .with_cluster_method(OpticsClusterMethod::Dbscan)
        .with_eps(f64::NAN)
        .fit(&x, &());
    // sklearn 1.5.2 oracle: InvalidParameterError (eps must be in [0.0, inf]).
    assert!(
        result.is_err(),
        "sklearn rejects eps=NaN (InvalidParameterError, _optics.py:252); \
         ferrolearn returned Ok"
    );
}

/// Companion: negative `eps` is likewise rejected by sklearn's `Interval(Real,
/// 0, None, closed="both")` (`_optics.py:252`) but accepted by ferrolearn — the
/// Dbscan arm's only `eps` guard is `eps > max_eps`, and `-1.0 > 5.0` is false.
///
/// Oracle: `OPTICS(min_samples=2, cluster_method='dbscan', eps=-1.0,
/// max_eps=5.0).fit(X)` raises `InvalidParameterError`. ferrolearn returns `Ok`.
///
/// Tracking: #2198
#[test]
#[ignore = "divergence: OPTICS accepts eps=NaN / eps<0; tracking #2198"]
fn divergence_eps_negative_dbscan_rejected() {
    let x = three_blobs();
    let result = OPTICS::<f64>::new(2)
        .with_max_eps(5.0)
        .with_cluster_method(OpticsClusterMethod::Dbscan)
        .with_eps(-1.0)
        .fit(&x, &());
    assert!(
        result.is_err(),
        "sklearn rejects eps<0 (InvalidParameterError, _optics.py:252); \
         ferrolearn returned Ok"
    );
}

/// Companion: sklearn validates `eps` even on the Xi path (the constraint is on
/// the estimator parameter, validated before the fit body, independent of
/// `cluster_method`). ferrolearn never reads `eps` on the Xi path, so a bad
/// `eps` is silently accepted.
///
/// Oracle: `OPTICS(min_samples=2, eps=nan).fit(X)` (default cluster_method='xi')
/// raises `InvalidParameterError`. ferrolearn returns `Ok`.
///
/// Tracking: #2198
#[test]
#[ignore = "divergence: OPTICS accepts eps=NaN / eps<0; tracking #2198"]
fn divergence_eps_nan_xi_rejected() {
    let x = three_blobs();
    let result = OPTICS::<f64>::new(2).with_eps(f64::NAN).fit(&x, &());
    assert!(
        result.is_err(),
        "sklearn rejects eps=NaN even on the Xi path (InvalidParameterError, \
         _optics.py:252); ferrolearn returned Ok"
    );
}
