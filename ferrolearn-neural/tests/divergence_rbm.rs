//! Adversarial divergence pins for `ferrolearn-neural/src/rbm.rs` (BernoulliRBM)
//! against scikit-learn 1.5.2 (`sklearn/neural_network/_rbm.py`, commit 156ef14).
//!
//! Expected values come from a LIVE sklearn 1.5.2 oracle invoked from /tmp
//! (R-CHAR-3): a `BernoulliRBM` whose `components_`, `intercept_hidden_`, and
//! `intercept_visible_` are set DIRECTLY (no `fit`, so the RNG is irrelevant),
//! then `_mean_hiddens` / `_free_energy` evaluated on a fixed binary `v`.
//!
//! These tests compile and run; the missing-API divergences (score_samples,
//! free_energy, partial_fit, h_samples_, n_features_in_) are pinned in the
//! companion file `divergence_rbm_missing_api.rs`, which fails to compile
//! until those methods/attributes ship.

use ferrolearn_neural::FittedBernoulliRBM;
use ndarray::{Array1, Array2, array};

/// Build a fitted RBM with KNOWN parameters (the same values fed to the live
/// sklearn oracle), bypassing `fit` so no RNG is involved.
fn known_fitted() -> FittedBernoulliRBM<f64> {
    let components_: Array2<f64> = array![[0.5, -0.3, 0.2, 0.1], [-0.2, 0.4, -0.1, 0.6],];
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

/// REQ-2 — `transform` value parity with sklearn `_mean_hiddens`
/// (`sklearn/neural_network/_rbm.py:193-195`:
/// `p = safe_sparse_dot(v, self.components_.T); p += self.intercept_hidden_; return expit(p, out=p)`).
///
/// Live sklearn 1.5.2 oracle (params set directly, no fit) on `fixed_v()`:
/// [[0.6899744811276125, 0.3775406687981454],
///  [0.47502081252106,   0.6899744811276125],
///  [0.6456563062257954, 0.6224593312018546]]
///
/// This test is EXPECTED TO PASS — the ferrolearn formula
/// `sigmoid(intercept_hidden_[j] + sum_k v[i,k]*components_[j,k])` is
/// arithmetically identical. It is recorded as a characterization pin
/// (R-CHAR-1) proving REQ-2's deterministic path does not diverge.
#[test]
fn divergence_transform_mean_hiddens_value_parity() {
    let fitted = known_fitted();
    let got = fitted.transform(&fixed_v()).expect("transform");
    // Live sklearn 1.5.2 `_mean_hiddens`.
    let expected: Array2<f64> = array![
        [0.6899744811276125, 0.3775406687981454],
        [0.47502081252106, 0.6899744811276125],
        [0.6456563062257954, 0.6224593312018546],
    ];
    assert_eq!(got.dim(), expected.dim());
    for (g, e) in got.iter().zip(expected.iter()) {
        assert!(
            (g - e).abs() < 1e-12,
            "transform diverges from sklearn _mean_hiddens: got {g}, sklearn {e}"
        );
    }
}
