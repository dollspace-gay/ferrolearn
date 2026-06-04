//! Missing-API divergence pins for `ferrolearn-neural/src/rbm.rs` (BernoulliRBM)
//! vs scikit-learn 1.5.2 (`sklearn/neural_network/_rbm.py`, commit 156ef14).
//!
//! sklearn's `BernoulliRBM` exposes `score_samples`, `_free_energy`,
//! `partial_fit`, and the fitted attributes `h_samples_` / `n_features_in_`.
//! ferrolearn's `FittedBernoulliRBM` exposes NONE of these. The absence is the
//! divergence: this test file FAILS TO COMPILE (method/field not found) until a
//! builder ships the API. Expected values are from the LIVE sklearn 1.5.2 oracle
//! (params set directly, no fit, RNG irrelevant) on the fixed binary `v` below
//! (R-CHAR-3).
//!
//! Each `#[ignore]`/tracking ref is recorded in the per-test doc comment; the
//! pins are deliberately un-ignored because the missing API is a release blocker
//! for REQ-7.

use ferrolearn_core::traits::Fit;
use ferrolearn_neural::{BernoulliRBM, FittedBernoulliRBM};
use ndarray::{Array1, Array2, array};

fn known_fitted() -> FittedBernoulliRBM<f64> {
    let components_: Array2<f64> = array![[0.5, -0.3, 0.2, 0.1], [-0.2, 0.4, -0.1, 0.6]];
    let intercept_hidden_: Array1<f64> = array![0.1, -0.2];
    let intercept_visible_: Array1<f64> = array![0.05, -0.15, 0.25, -0.35];
    FittedBernoulliRBM {
        components_,
        intercept_hidden_,
        intercept_visible_,
        h_samples_: Array2::zeros((0, 2)),
        n_iter_: 0,
    }
}

fn fixed_v() -> Array2<f64> {
    array![
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
    ]
}

/// REQ-7 — `_free_energy` is missing.
/// sklearn `sklearn/neural_network/_rbm.py:250-252`:
/// `return -safe_sparse_dot(v, self.intercept_visible_) - np.logaddexp(0,
///  safe_sparse_dot(v, self.components_.T) + self.intercept_hidden_).sum(axis=1)`.
/// Live sklearn 1.5.2 oracle on `fixed_v()` with the known params:
/// [-1.9451776501278844, -1.3154973260213487, -1.8115649346659926].
/// Tracking: free-energy blocker referenced under #1628.
#[test]
fn divergence_free_energy_missing() {
    let fitted = known_fitted();
    let expected: Array1<f64> = array![
        -1.9451776501278844,
        -1.3154973260213487,
        -1.8115649346659926
    ];
    // METHOD DOES NOT EXIST → compile failure = the pinned divergence.
    let got = fitted.free_energy(&fixed_v()).expect("free_energy");
    for (g, e) in got.iter().zip(expected.iter()) {
        assert!((g - e).abs() < 1e-12, "free_energy diverges: {g} vs {e}");
    }
}

/// REQ-7 — `score_samples` (pseudo-likelihood) is missing.
/// sklearn `sklearn/neural_network/_rbm.py:383-386`:
/// `fe = self._free_energy(v); fe_ = self._free_energy(v_);
///  return -v.shape[1] * np.logaddexp(0, -(fe_ - fe))` where one feature per row
/// is corrupted (`_rbm.py:372,381`). With a FIXED corruption index [0,1,2] the
/// arithmetic is deterministic; live sklearn 1.5.2 oracle yields:
/// [-2.240546579745612, -2.8649444947448055, -2.198651158575029].
/// (The RNG-drawn corruption index itself is an R-DEFER-3 carve-out; only the
/// free-energy arithmetic for a fixed pattern is pinned here.)
/// Tracking: score_samples blocker referenced under #1628.
#[test]
fn divergence_score_samples_missing() {
    let fitted = known_fitted();
    let corrupt_idx: Vec<usize> = vec![0, 1, 2];
    let expected: Array1<f64> = array![-2.240546579745612, -2.8649444947448055, -2.198651158575029];
    // METHOD DOES NOT EXIST → compile failure = the pinned divergence.
    let got = fitted
        .score_samples_with_corruption(&fixed_v(), &corrupt_idx)
        .expect("score_samples");
    for (g, e) in got.iter().zip(expected.iter()) {
        assert!((g - e).abs() < 1e-12, "score_samples diverges: {g} vs {e}");
    }
}

/// REQ-7 — persistent particle attribute `h_samples_` is missing.
/// sklearn keeps `self.h_samples_` of shape `(batch_size, n_components)`
/// (`sklearn/neural_network/_rbm.py:417`:
/// `self.h_samples_ = np.zeros((self.batch_size, self.n_components), ...)`),
/// the locus of SML/PCD persistent particles (REQ-3, `_rbm.py:332,344-345`).
/// ferrolearn's `FittedBernoulliRBM` has no such field; the negative phase
/// samples from the current batch instead. Structural pin for the SML
/// divergence: this references `fitted.h_samples_` which does not exist.
/// Tracking: SML/PCD + h_samples_ blocker referenced under #1628.
#[test]
fn divergence_h_samples_attribute_missing() {
    let x: Array2<f64> = array![
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
    ];
    let fitted = BernoulliRBM::<f64>::new(5)
        .batch_size(2)
        .n_iter(1)
        .random_state(0)
        .fit(&x, &())
        .expect("fit");
    // FIELD DOES NOT EXIST → compile failure = the pinned divergence.
    // sklearn: h_samples_.shape == (batch_size, n_components) == (2, 5).
    assert_eq!(fitted.h_samples_.dim(), (2, 5));
}

/// REQ-7 — `partial_fit` (incremental SML fit) is missing.
/// sklearn `sklearn/neural_network/_rbm.py:276-315`: `partial_fit` lazily
/// initializes `components_`/intercepts/`h_samples_` on first call and runs one
/// `_fit` step, enabling out-of-core/incremental training. ferrolearn's
/// `BernoulliRBM` has no `partial_fit`. This references the missing method.
/// Tracking: partial_fit blocker referenced under #1628.
#[test]
fn divergence_partial_fit_missing() {
    let x: Array2<f64> = array![[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]];
    let mut rbm = BernoulliRBM::<f64>::new(3).batch_size(2).random_state(0);
    // METHOD DOES NOT EXIST → compile failure = the pinned divergence.
    let fitted = rbm.partial_fit(&x, &()).expect("partial_fit");
    assert_eq!(fitted.components_.dim(), (3, 4));
}
