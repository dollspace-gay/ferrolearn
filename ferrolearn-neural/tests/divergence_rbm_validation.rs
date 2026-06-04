//! Parameter-validation divergence pins for `ferrolearn-neural/src/rbm.rs`
//! (BernoulliRBM) vs scikit-learn 1.5.2.
//!
//! sklearn `_parameter_constraints` (`sklearn/neural_network/_rbm.py:134-141`):
//!   "n_components": [Interval(Integral, 1, None, closed="left")],      # >= 1
//!   "learning_rate": [Interval(Real, 0, None, closed="neither")],      # > 0
//!   "batch_size": [Interval(Integral, 1, None, closed="left")],        # >= 1
//!   "n_iter": [Interval(Integral, 0, None, closed="left")],            # >= 0
//! Live sklearn 1.5.2 oracle: `BernoulliRBM(n_components=0).fit(X)`,
//! `learning_rate=0.0`, `batch_size=0`, `n_iter=-1` each raise
//! `InvalidParameterError` (a `ValueError` subclass).
//!
//! ferrolearn `BernoulliRBM::new(0)` and `.learning_rate(0.0)` are accepted, and
//! `batch_size(0)` is silently clamped to 1 (`fn batch_size`: `n.max(1)`); `fit`
//! does NOT reject them. These tests assert `fit` returns `Err`, so they FAIL
//! against the current implementation.
//!
//! Tracking: parameter-validation blocker referenced under #1628.

use ferrolearn_core::traits::Fit;
use ferrolearn_neural::BernoulliRBM;
use ndarray::{Array2, array};

fn x() -> Array2<f64> {
    array![[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]
}

/// REQ-1 — `n_components < 1` must be rejected.
/// sklearn: `Interval(Integral, 1, None, closed="left")` (`_rbm.py:135`) →
/// `BernoulliRBM(n_components=0).fit(X)` raises InvalidParameterError.
#[test]
fn divergence_n_components_zero_rejected() {
    let rbm = BernoulliRBM::<f64>::new(0);
    assert!(
        rbm.fit(&x(), &()).is_err(),
        "sklearn rejects n_components=0 (InvalidParameterError); ferrolearn accepts it"
    );
}

/// REQ-1 — `learning_rate <= 0` must be rejected.
/// sklearn: `Interval(Real, 0, None, closed="neither")` (`_rbm.py:136`) →
/// `BernoulliRBM(learning_rate=0.0).fit(X)` raises InvalidParameterError.
#[test]
fn divergence_learning_rate_zero_rejected() {
    let rbm = BernoulliRBM::<f64>::new(2).learning_rate(0.0);
    assert!(
        rbm.fit(&x(), &()).is_err(),
        "sklearn rejects learning_rate=0.0 (InvalidParameterError); ferrolearn accepts it"
    );
}

/// REQ-1 — `batch_size < 1` must be rejected, not silently clamped.
/// sklearn: `Interval(Integral, 1, None, closed="left")` (`_rbm.py:137`) →
/// `BernoulliRBM(batch_size=0).fit(X)` raises InvalidParameterError. ferrolearn
/// `fn batch_size` does `n.max(1)`, silently clamping 0 -> 1 (a divergence: a
/// caller who asked for batch_size=0 gets a fit, not an error).
#[test]
fn divergence_batch_size_zero_rejected() {
    let rbm = BernoulliRBM::<f64>::new(2).batch_size(0);
    assert!(
        rbm.fit(&x(), &()).is_err(),
        "sklearn rejects batch_size=0 (InvalidParameterError); ferrolearn clamps to 1 and fits"
    );
}
