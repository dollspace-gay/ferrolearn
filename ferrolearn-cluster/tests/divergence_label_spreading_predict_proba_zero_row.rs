//! Divergence pin: `FittedLabelSpreading::predict_proba` zero-row handling vs
//! scikit-learn 1.5.2 `BaseLabelPropagation.predict_proba`
//! (`sklearn/semi_supervised/_label_propagation.py:229-230`).
//!
//! sklearn's `predict_proba` row-normalizes the kernel-weighted combination with
//! an UNGUARDED divide (`:229-230`):
//! ```python
//! normalizer = np.atleast_2d(np.sum(probabilities, axis=1)).T
//! probabilities /= normalizer
//! ```
//! There is NO `normalizer[normalizer == 0] = 1` zero-guard here — that guard
//! exists only inside `fit` (`:328-330`). When a query point is far enough from
//! every training row that all RBF weights underflow to `0`, the row sum is `0`
//! and sklearn emits `nan` for that row (live oracle below). ferrolearn's
//! `predict_proba` instead guards with `if row_sum > F::zero()`
//! (`label_spreading.rs:360`), leaving the row at finite `[0.0, 0.0]`. The
//! observable outputs diverge: sklearn `[nan, nan]`, ferrolearn `[0.0, 0.0]`.
//!
//! LIVE sklearn 1.5.2 oracle (R-CHAR-3 — expected value computed from sklearn,
//! NOT copied from ferrolearn):
//! ```text
//! from sklearn.semi_supervised import LabelSpreading
//! m = LabelSpreading(gamma=1, alpha=0.2)
//! X = [[0,0],[0.3,0],[0.6,0],[1,0]]; y = [0,-1,-1,1]
//! m.fit(X, y)
//! m.predict_proba([[1e6, 1e6]])   # -> [[nan, nan]]  (RuntimeWarning: invalid value in divide)
//! np.isnan(...)                    # -> [[True, True]]
//! ```
//! sklearn delegates the kernel to `rbf_kernel` (`:147`) and normalizes at
//! `:229-230`; the resulting row is NOT finite.
//!
//! Tracking: #2184.

use ferrolearn_cluster::LabelSpreading;
use ferrolearn_core::Fit;
use ndarray::{Array1, Array2};

/// sklearn `predict_proba` produces a NON-FINITE (`nan`) row for a query point
/// far from all training rows (unguarded `/=` at `:229-230`); ferrolearn returns
/// a finite `[0,0]` row. This test asserts the sklearn observable (the row is NOT
/// finite) and therefore FAILS against the current ferrolearn guard.
///
/// Tracking: #2184.
#[test]
fn divergence_predict_proba_zero_row_is_nan_not_finite() {
    // Same fixture the live oracle was run on (`line`, gamma=1, alpha=0.2).
    let x = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.3, 0.0, 0.6, 0.0, 1.0, 0.0]).unwrap();
    let y = Array1::from_vec(vec![0, -1, -1, 1]);

    let fitted = LabelSpreading::<f64>::new()
        .with_gamma(1.0)
        .fit(&x, &y)
        .unwrap();

    // Query point far from every training row: all RBF weights underflow to 0,
    // so the unnormalized probability row is [0, 0] and sklearn's unguarded
    // `probabilities /= normalizer` yields [nan, nan].
    let xq = Array2::from_shape_vec((1, 2), vec![1.0e6, 1.0e6]).unwrap();
    let proba = fitted.predict_proba(&xq).unwrap();

    // sklearn oracle: m.predict_proba([[1e6,1e6]]) -> [[nan, nan]] (both NON-finite).
    // ferrolearn's zero-row guard returns finite [0.0, 0.0], so these asserts fail.
    let p0 = proba[[0, 0]];
    let p1 = proba[[0, 1]];
    assert!(
        !p0.is_finite() && !p1.is_finite(),
        "sklearn predict_proba(:229-230) emits a NON-finite (nan) row for a \
         far-away query (row sum 0, unguarded /=); ferrolearn returned finite \
         [{p0}, {p1}]. Divergence: ferrolearn guards the zero row, sklearn does not."
    );
}
