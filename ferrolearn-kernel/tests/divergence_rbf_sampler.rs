//! Divergence + green-guard tests for `ferrolearn-kernel::RBFSampler` against
//! scikit-learn 1.5.2 `sklearn.kernel_approximation.RBFSampler`
//! (`sklearn/kernel_approximation.py:244-411`).
//!
//! Tracking: #1669. Per-REQ blockers: #1670 (gamma=0), #1671 (prod unwrap),
//! #1672 (RNG carve-out), #1673 (gamma='scale'), #1674 (n_features_in_),
//! #1675 (ferray substrate).
//!
//! Expected values come from the live sklearn 1.5.2 oracle (run from /tmp) or
//! the sklearn transform/sampling FORMULA — NEVER copied from ferrolearn's own
//! transform output (goal.md R-CHAR-3).

use ferrolearn_core::{Fit, Transform};
use ferrolearn_kernel::RBFSampler;
use ndarray::{Array2, array};

// ---------------------------------------------------------------------------
// FAILING test — REQ-3: gamma=0 over-rejection.
// ---------------------------------------------------------------------------

/// Divergence: ferrolearn's `RBFSampler::fit` diverges from
/// `sklearn/kernel_approximation.py:330`
/// (`gamma: Interval(Real, 0.0, None, closed="left")`) for `gamma=0`.
///
/// sklearn ACCEPTS `gamma=0` (live oracle from /tmp:
/// `RBFSampler(gamma=0.0, n_components=3, random_state=0).fit([[1,2],[3,4]])`
/// succeeds with `random_weights_` all-zero, since `(2.0*0.0)**0.5 == 0.0`).
/// ferrolearn `fn fit` (`rbf_sampler.rs:143`) returns
/// `Err(InvalidParameter{ reason: "must be positive" })`.
///
/// sklearn returns: Ok with all-zero `random_weights_`.
/// ferrolearn returns: Err(InvalidParameter).
///
/// NOTE for the fixer: the in-crate unit test `rejects_zero_gamma`
/// (`rbf_sampler.rs:371-376`) asserts the CURRENT (divergent) behavior and must
/// be removed/inverted when REQ-3 lands. `rejects_negative_gamma` is correct
/// (sklearn rejects gamma<0) and should stay.
///
/// Tracking: #1670
#[test]
#[ignore = "divergence: RBFSampler rejects gamma=0; sklearn accepts it; tracking #1670"]
fn divergence_gamma_zero_accepted() {
    let x: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
    let fitted = RBFSampler::<f64>::new()
        .with_gamma(0.0)
        .with_n_components(3)
        .with_random_state(0)
        .fit(&x, &());

    // sklearn accepts gamma=0 -> ferrolearn should too.
    assert!(
        fitted.is_ok(),
        "sklearn RBFSampler(gamma=0.0).fit succeeds (oracle), ferrolearn must too; got {:?}",
        fitted.err()
    );

    // and the resulting weights are all zero (sqrt(2*0) == 0).
    let fitted = fitted.unwrap();
    for &w in fitted.random_weights().iter() {
        assert_eq!(w, 0.0, "sklearn gamma=0 yields all-zero random_weights_");
    }
}

// ---------------------------------------------------------------------------
// GREEN guard — REQ-1: transform formula parity.
// ---------------------------------------------------------------------------

/// Green guard: `FittedRBFSampler::transform` mirrors the sklearn transform
/// FORMULA `sqrt(2/n_components)*cos(X·W + b)`
/// (`sklearn/kernel_approximation.py:404-407`).
///
/// R-CHAR-3-compliant: the expected value is recomputed in-test from the
/// sklearn formula using ferrolearn's OWN fitted `random_weights()`/
/// `random_offset()` as inputs — it is NOT copied from `transform`'s output.
/// This pins the deterministic mapping without an RNG bit-match.
///
/// Tracking: #1669 (REQ-1, SHIPPED).
#[test]
fn green_transform_formula_parity() {
    // Non-trivial 6x3 input, n_components = 5 (>= 4).
    let x: Array2<f64> = array![
        [0.5, -1.2, 3.4],
        [2.1, 0.0, -0.7],
        [-1.5, 2.2, 1.1],
        [3.3, -2.8, 0.9],
        [0.1, 1.7, -3.0],
        [-2.4, 0.6, 2.5],
    ];
    let n_components = 5usize;
    let fitted = RBFSampler::<f64>::new()
        .with_gamma(0.7)
        .with_n_components(n_components)
        .with_random_state(5)
        .fit(&x, &())
        .expect("fit should succeed");

    let w = fitted.random_weights();
    let b = fitted.random_offset();
    let scale = (2.0 / n_components as f64).sqrt();

    // Expected from sklearn formula :404-407 using ferrolearn's own W, b.
    let proj = x.dot(w) + b;
    let expected = proj.mapv(|v| scale * v.cos());

    let actual = fitted.transform(&x).expect("transform should succeed");
    assert_eq!(actual.dim(), expected.dim());

    for (a, e) in actual.iter().zip(expected.iter()) {
        assert!(
            (a - e).abs() <= 1e-12,
            "transform diverges from sklearn formula: actual={a}, expected={e}"
        );
    }
}

// ---------------------------------------------------------------------------
// GREEN guard — REQ-2: sampling-distribution structure.
// ---------------------------------------------------------------------------

/// Green guard: `fit` draws `random_weights_` of shape
/// `(n_features, n_components)` with column std ≈ `sqrt(2*gamma)` and
/// `random_offset_` of shape `(n_components,)` in `[0, 2*pi)`
/// (`sklearn/kernel_approximation.py:372-376`).
///
/// Expected std is the sklearn FORMULA `sqrt(2*gamma)` (`:372`), not a
/// ferrolearn output. Oracle confirms (from /tmp): for gamma=2, large
/// n_components, `random_weights_.std() -> ~sqrt(2*2)=2.0` and offsets in
/// `[0, 2*pi)`.
///
/// Tracking: #1669 (REQ-2, SHIPPED).
#[test]
fn green_sampling_structure() {
    let n_features = 4usize;
    let n_components = 20000usize;
    let gamma = 2.0f64;
    let x: Array2<f64> = Array2::zeros((3, n_features));

    let fitted = RBFSampler::<f64>::new()
        .with_gamma(gamma)
        .with_n_components(n_components)
        .with_random_state(7)
        .fit(&x, &())
        .expect("fit should succeed");

    let w = fitted.random_weights();
    let b = fitted.random_offset();

    // Shapes (sklearn :273-280).
    assert_eq!(w.dim(), (n_features, n_components), "random_weights_ shape");
    assert_eq!(b.len(), n_components, "random_offset_ shape");

    // Offsets in [0, 2*pi) (sklearn :376 uniform(0, 2*pi)).
    let two_pi = 2.0 * std::f64::consts::PI;
    for &v in b.iter() {
        assert!((0.0..two_pi).contains(&v), "offset {v} outside [0, 2*pi)");
    }

    // Empirical std ≈ sqrt(2*gamma) (sklearn formula :372).
    let n = (n_features * n_components) as f64;
    let mean = w.iter().sum::<f64>() / n;
    let var = w.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n;
    let std = var.sqrt();
    let expected_std = (2.0 * gamma).sqrt();
    assert!(
        (std - expected_std).abs() < 0.05,
        "weight std {std} should approximate sklearn sqrt(2*gamma)={expected_std}"
    );
}

// ---------------------------------------------------------------------------
// GREEN guard — REQ-6: n_components=0 validation.
// ---------------------------------------------------------------------------

/// Green guard: ferrolearn rejects `n_components=0`, matching sklearn's
/// `_parameter_constraints`
/// (`sklearn/kernel_approximation.py:332`,
/// `n_components: Interval(Integral, 1, None, closed="left")`).
///
/// Oracle (from /tmp): `RBFSampler(n_components=0).fit(X)` raises
/// `InvalidParameterError`. Both reject -> green.
///
/// Tracking: #1669 (REQ-6, SHIPPED).
#[test]
fn green_n_components_zero_rejected() {
    let x: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
    let fitted = RBFSampler::<f64>::new()
        .with_n_components(0)
        .with_random_state(0)
        .fit(&x, &());
    assert!(
        fitted.is_err(),
        "sklearn raises InvalidParameterError for n_components=0; ferrolearn must reject too"
    );
}
