//! Re-audit of the SML/PCD rebuild (commit 0dbe4723) of `BernoulliRBM`.
//!
//! Independent parity pins for the DETERMINISTIC methods, derived from a FRESH
//! live scikit-learn 1.5.2 oracle call with components/intercepts and input
//! data DISTINCT from the builder's own `known_fitted()`/`fixed_v()` fixtures
//! (R-CHAR-3: expected values come from the oracle, never copied from
//! ferrolearn). A 3-component, 5-feature, 4-sample case with non-trivial,
//! asymmetric components.
//!
//! Oracle script (`/tmp`, sklearn 1.5.2):
//! ```python
//! rbm = BernoulliRBM(n_components=3)
//! rbm.components_ = [[0.7,-0.4,0.2,0.1,-0.5],[-0.3,0.6,-0.2,0.4,0.15],
//!                    [0.05,-0.25,0.35,-0.45,0.55]]
//! rbm.intercept_hidden_  = [0.2,-0.3,0.1]
//! rbm.intercept_visible_ = [0.1,-0.2,0.3,-0.4,0.05]
//! v = [[1,0,1,0,1],[0,1,0,1,0],[1,1,0,0,1],[0,1,1,1,0]]
//! ```

use ferrolearn_neural::rbm::FittedBernoulliRBM;
use ndarray::{Array2, array};

fn known_fitted() -> FittedBernoulliRBM<f64> {
    FittedBernoulliRBM {
        components_: array![
            [0.7, -0.4, 0.2, 0.1, -0.5],
            [-0.3, 0.6, -0.2, 0.4, 0.15],
            [0.05, -0.25, 0.35, -0.45, 0.55]
        ],
        intercept_hidden_: array![0.2, -0.3, 0.1],
        intercept_visible_: array![0.1, -0.2, 0.3, -0.4, 0.05],
        h_samples_: Array2::zeros((0, 3)),
        n_iter_: 0,
    }
}

fn fixed_v() -> Array2<f64> {
    array![
        [1.0, 0.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0, 1.0, 0.0],
    ]
}

/// `BernoulliRBM::free_energy` vs sklearn `_rbm.py:250-252`
/// (`-(v@b_v) - sum_j logaddexp(0, (v@W.T+b_h))`), fresh oracle.
#[test]
fn reaudit_free_energy_matches_sklearn() {
    let got = known_fitted().free_energy(&fixed_v()).unwrap();
    let expected = [
        -3.2576017658062444,
        -1.5850706594449142,
        -2.3573531743469323,
        -1.994413064132521,
    ];
    for (g, e) in got.iter().zip(expected.iter()) {
        assert!((g - e).abs() < 1e-10, "free_energy: {g} vs {e}");
    }
}

/// `transform` vs sklearn `_mean_hiddens` (`_rbm.py:193-195`), fresh oracle.
#[test]
fn reaudit_transform_matches_mean_hiddens() {
    let got = known_fitted().transform(&fixed_v()).unwrap();
    let expected = array![
        [0.64565631, 0.34298954, 0.7407749],
        [0.47502081, 0.66818777, 0.35434369],
        [0.5, 0.53742985, 0.61063923],
        [0.52497919, 0.62245933, 0.4378235],
    ];
    for (g, e) in got.iter().zip(expected.iter()) {
        assert!((g - e).abs() < 1e-8, "mean_hiddens: {g} vs {e}");
    }
}

/// `score_samples_with_corruption` vs sklearn `score_samples` arithmetic
/// (`_rbm.py:383-386`) with the SAME explicit corruption indices, fresh oracle.
#[test]
fn reaudit_score_samples_with_corruption_matches_sklearn() {
    let got = known_fitted()
        .score_samples_with_corruption(&fixed_v(), &[0, 2, 4, 1])
        .unwrap();
    let expected = [
        -2.5320096292629604,
        -4.593094566829831,
        -3.1146669980617423,
        -4.038184527660359,
    ];
    for (g, e) in got.iter().zip(expected.iter()) {
        assert!((g - e).abs() < 1e-10, "score_samples: {g} vs {e}");
    }
}

// ---------------------------------------------------------------------------
// DIVERGENCE: partial_fit is not incremental.
// ---------------------------------------------------------------------------

use ferrolearn_neural::rbm::BernoulliRBM;

/// Divergence: `ferrolearn-neural`'s `BernoulliRBM::partial_fit`
/// (`ferrolearn-neural/src/rbm.rs:159-199`) diverges from sklearn
/// `BernoulliRBM.partial_fit` (`sklearn/neural_network/_rbm.py:292-315`).
///
/// sklearn line 292: `first_pass = not hasattr(self, "components_")` — the
/// `components_`/`intercept_*_`/`h_samples_` are initialized ONLY on the first
/// call; every subsequent `partial_fit` runs `self._fit(X, ...)` (`:315`)
/// against the EXISTING, accumulated parameters and persistent particles. Two
/// successive `partial_fit` calls on the same X (seed fixed) therefore yield
/// DIFFERENT `components_` (the second continues from the first's non-zero
/// state).
///
/// ferrolearn's `partial_fit` instead re-initializes `components` from
/// `N(0, 0.01)`, zeroes the intercepts and `h_samples`, and returns a fresh
/// `FittedBernoulliRBM` on EVERY call (`rbm.rs:175-198`); it never reads prior
/// state. With a fixed seed, call 1 and call 2 produce byte-identical output —
/// no accumulation. This is not the documented RNG carve-out (the divergence
/// persists for any fixed seed and is about *which state the update is applied
/// to*, not the RNG stream).
///
/// Oracle (`/tmp`, sklearn 1.5.2): with `random_state=0`, `batch_size=2`,
/// X = [[1,0,1,0],[0,1,0,1]], the second `partial_fit` changes `components_`
/// (`not np.allclose(c1, c2)` == True).
///
/// Tracking: #1639
#[test]
fn divergence_partial_fit_is_not_incremental() {
    let mut rbm = BernoulliRBM::<f64>::new(3)
        .batch_size(2)
        .random_state(0)
        .learning_rate(0.1);
    let x = array![[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]];

    let f1 = rbm.partial_fit(&x, &()).unwrap();
    let f2 = rbm.partial_fit(&x, &()).unwrap();

    // sklearn: the second incremental step CONTINUES from the first, so the
    // two component matrices differ. ferrolearn re-inits identically => they
    // are bit-equal, which this assertion forbids.
    let max_diff = (&f1.components_ - &f2.components_)
        .iter()
        .fold(0.0_f64, |m, &d| m.max(d.abs()));
    assert!(
        max_diff > 1e-12,
        "partial_fit must accumulate: a 2nd call should change components_ \
         (sklearn _rbm.py:315), but ferrolearn re-inits identically \
         (max |c1-c2| = {max_diff})"
    );
}
