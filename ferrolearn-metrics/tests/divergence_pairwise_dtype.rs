//! Divergence pins for `ferrolearn-metrics/src/pairwise.rs` vs scikit-learn
//! 1.5.2 — the f32-upcast and NaN/Inf-validation paths that the existing
//! `divergence_pairwise.rs` GREEN guards (f64-only) do NOT cover.
//!
//! All expected values / exceptions come from the LIVE sklearn 1.5.2 oracle
//! (the exact `python3 -c` call is quoted per test, R-CHAR-3) — NEVER copied
//! from the ferrolearn side.
//!
//! These are genuine VALUE / EXCEPTION divergences (R-DEV-1 numerical parity,
//! R-DEV-2 exception parity), distinct from the design doc's acknowledged
//! missing-surface ABI gaps.

use ferrolearn_metrics::pairwise::{cosine_distances, euclidean_distances};
use ndarray::array;

// ===========================================================================
// f32 catastrophic cancellation — sklearn upcasts f32 chunks to f64 for the
// `||x||^2 + ||y||^2 - 2*x.y` trick (`sklearn/metrics/pairwise.py:401-404`,
// `_euclidean_distances_upcast`). ferrolearn computes the trick NATIVELY in f32
// (`pairwise.rs:195-208`), so the large-magnitude squared terms lose their low
// bits, the difference cancels to a (clamped) 0, and the true distance is lost.
// Tracking: #2298
// ===========================================================================

/// Divergence: `euclidean_distances([[1e5_f32]], [[100008_f32]])`.
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.metrics.pairwise import \
///     euclidean_distances as e; \
///     print(repr(e(np.array([[1e5]],np.float32), \
///       np.array([[100008.]],np.float32)).ravel()[0]))"
///   # 8.0
///
/// sklearn returns `8.0`; ferrolearn returns `0.0`.
#[test]
#[ignore = "divergence: euclidean f32 native-trick cancellation vs sklearn f64 upcast; tracking #2298"]
fn divergence_euclidean_f32_cancellation_single() {
    let x = array![[1e5_f32]];
    let y = array![[100008.0_f32]];
    let d = euclidean_distances(&x, &y).unwrap();
    // sklearn 1.5.2 live oracle: 8.0.
    const SK: f32 = 8.0;
    assert!(
        (d[[0, 0]] - SK).abs() < 1e-2,
        "euclidean f32: sklearn={SK}, ferrolearn={}",
        d[[0, 0]]
    );
}

/// Divergence: `euclidean_distances([[1e5,2e5]], [[1e5,200000.1]])` (f32).
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.metrics.pairwise import \
///     euclidean_distances as e; \
///     print(repr(e(np.array([[1e5,2e5]],np.float32), \
///       np.array([[1e5,200000.1]],np.float32)).ravel()[0]))"
///   # 0.09375
///
/// sklearn returns `0.09375`; ferrolearn returns `0.0`.
#[test]
#[ignore = "divergence: euclidean f32 native-trick cancellation vs sklearn f64 upcast; tracking #2298"]
fn divergence_euclidean_f32_cancellation_multifeature() {
    let x = array![[100000.0_f32, 200000.0]];
    let y = array![[100000.0_f32, 200000.1]];
    let d = euclidean_distances(&x, &y).unwrap();
    // sklearn 1.5.2 live oracle: 0.09375.
    const SK: f32 = 0.09375;
    assert!(
        (d[[0, 0]] - SK).abs() < 1e-3,
        "euclidean f32 multifeature: sklearn={SK}, ferrolearn={}",
        d[[0, 0]]
    );
}

// ===========================================================================
// NaN / Inf input validation — sklearn's `check_pairwise_arrays` defaults
// `force_all_finite=True` (`sklearn/metrics/pairwise.py:86`, invoked at `:344`
// for euclidean and via `cosine_similarity` for cosine) and raises a
// `ValueError` on any non-finite entry. ferrolearn performs NO finiteness
// validation and, because Rust's `NaN.max(0.0) == 0.0`, SILENTLY returns a
// finite WRONG value (`0.0`, indistinguishable from identical points).
// Tracking: #2299
// ===========================================================================

/// Divergence: `euclidean_distances` on NaN input.
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.metrics.pairwise import \
///     euclidean_distances as e; \
///     e(np.array([[np.nan,0.]]), np.array([[1.,1.]]))"
///   # ValueError: Input contains NaN.
///
/// Parity contract: ferrolearn must NOT silently return a finite value. It
/// currently returns `Ok(0.0)`.
#[test]
fn divergence_euclidean_nan_input() {
    let x = array![[f64::NAN, 0.0_f64]];
    let y = array![[1.0_f64, 1.0]];
    let d = euclidean_distances(&x, &y);
    // sklearn raises ValueError("Input contains NaN."); ferrolearn must not
    // return a finite, plausible-looking value.
    let returned_finite = matches!(d, Ok(ref m) if m[[0, 0]].is_finite());
    assert!(
        !returned_finite,
        "sklearn raises ValueError(Input contains NaN); ferrolearn returned Ok({:?})",
        d.map(|m| m[[0, 0]])
    );
}

/// Divergence: `cosine_distances` on NaN input.
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.metrics.pairwise import \
///     cosine_distances as c; \
///     c(np.array([[np.nan,0.]]), np.array([[1.,1.]]))"
///   # ValueError: Input contains NaN.
///
/// ferrolearn currently returns `Ok(0.0)`.
#[test]
fn divergence_cosine_nan_input() {
    let x = array![[f64::NAN, 0.0_f64]];
    let y = array![[1.0_f64, 1.0]];
    let d = cosine_distances(&x, &y);
    let returned_finite = matches!(d, Ok(ref m) if m[[0, 0]].is_finite());
    assert!(
        !returned_finite,
        "sklearn raises ValueError(Input contains NaN); ferrolearn returned Ok({:?})",
        d.map(|m| m[[0, 0]])
    );
}

/// Divergence: `euclidean_distances` on infinite input.
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.metrics.pairwise import \
///     euclidean_distances as e; \
///     e(np.array([[np.inf,0.]]), np.array([[1.,1.]]))"
///   # ValueError: Input contains infinity or a value too large for dtype('float64').
///
/// ferrolearn currently returns `Ok(0.0)`.
#[test]
fn divergence_euclidean_inf_input() {
    let x = array![[f64::INFINITY, 0.0_f64]];
    let y = array![[1.0_f64, 1.0]];
    let d = euclidean_distances(&x, &y);
    let returned_finite = matches!(d, Ok(ref m) if m[[0, 0]].is_finite());
    assert!(
        !returned_finite,
        "sklearn raises ValueError(Input contains infinity); ferrolearn returned Ok({:?})",
        d.map(|m| m[[0, 0]])
    );
}
