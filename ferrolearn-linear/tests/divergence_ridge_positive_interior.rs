//! Parity audit of `ferrolearn_linear::Ridge` with `positive=true` at an
//! INTERIOR optimum, against the live scikit-learn 1.5.2 oracle.
//!
//! Context: the `positive=true` path was surfaced through the PyO3 binding +
//! Python wrappers (#2129). Active-constraint fixtures (a coefficient clamped to
//! 0) match sklearn to machine epsilon, because the binding correctly threads
//! `positive` and the vertex solution is uniquely determined. This file covers
//! the INTERIOR optimum: the unconstrained ridge solution is already
//! all-positive, so `positive=true` is a NO-OP constraint. By strict convexity +
//! KKT the feasible unconstrained minimizer IS the constrained minimizer, so
//! `solve_nonneg_ridge` (`with_positive` -> REQ-9 #387/#2131) short-circuits to
//! the exact closed-form unconstrained Cholesky optimum.
//!
//! R-DEV-6: sklearn's DEFAULT `Ridge(positive=True)` L-BFGS-B
//! (`sklearn/linear_model/_ridge.py:923-928` dispatch -> `_solve_lbfgs`
//! `_ridge.py:300`) runs at `tol=1e-4` and UNDER-CONVERGES on interior optima by
//! ~1.8e-4; at `tol=1e-12` it reaches the true optimum, which equals ferrolearn's
//! exact Cholesky optimum (to ~1e-11). ferrolearn ships that TRUE optimum, so it
//! matches a CONVERGED sklearn and is at least as precise as the default.
//!
//! All expected values are computed by live-calling a CONVERGED sklearn 1.5.2
//! (`tol=1e-12`) (R-CHAR-3); no value is literal-copied from the ferrolearn side.
//!
//! Tracking: #2131 (#387/#412 lineage). The #2129 BINDING SURFACE is correct.

use ferrolearn_core::introspection::HasCoefficients;
use ferrolearn_core::traits::Fit;
use ferrolearn_linear::Ridge;
use ndarray::{Array1, Array2, array};

/// 6x2 design whose unconstrained Ridge(alpha=0.1) optimum is all-positive
/// (interior): `Ridge(alpha=0.1).coef_ == [0.87211956, 1.95199882]`, both > 0,
/// so `positive=True` is an inactive constraint.
fn interior_fixture() -> (Array2<f64>, Array1<f64>) {
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![1.0, 0.2, 0.3, 1.0, 0.5, 0.5, 1.2, 0.1, 0.1, 1.3, 0.8, 0.9],
    )
    .unwrap();
    // y = X @ [2.0, 3.0] exactly.
    let y = array![
        2.6,
        3.6,
        2.5,
        2.7,
        4.100_000_000_000_000_5,
        4.300_000_000_000_001
    ];
    (x, y)
}

/// Parity: `ferrolearn_linear::Ridge(alpha=0.1, positive=true)` matches a
/// CONVERGED sklearn (`tol=1e-12`) at an INTERIOR optimum
/// (`sklearn/linear_model/_ridge.py:923-928` dispatch -> `_solve_lbfgs`
/// `_ridge.py:300`).
///
/// Live sklearn 1.5.2 oracle (CONVERGED, tol=1e-12):
/// ```text
/// python3 -c "import numpy as np; from sklearn.linear_model import Ridge; \
///   X=np.array([[1.,0.2],[0.3,1.],[0.5,0.5],[1.2,0.1],[0.1,1.3],[0.8,0.9]]); \
///   y=X@np.array([2.,3.]); \
///   print(Ridge(alpha=0.1,positive=True,tol=1e-12,max_iter=100000).fit(X,y).coef_.tolist())"
/// # -> [0.8721195612110347, 1.951998822043437]
/// # unconstrained Ridge(alpha=0.1).coef_ -> [0.8721195612162265, 1.9519988220569833]
/// #   (all > 0; converged positive == unconstrained to ~1e-11)
/// ```
/// R-DEV-6: sklearn's DEFAULT `Ridge(positive=True)` runs L-BFGS-B at `tol=1e-4`
/// and UNDER-CONVERGES on this interior optimum, stopping at
/// `[0.8719358921480487, 1.9518078349228183]` — ~1.8e-4 short of the true
/// optimum. At `tol=1e-12` it reaches the true optimum, which equals
/// ferrolearn's exact closed-form unconstrained Cholesky optimum (the constraint
/// is inactive). ferrolearn ships that TRUE optimum (the inactive-constraint
/// short-circuit in `solve_nonneg_ridge`), so it matches the CONVERGED sklearn
/// and is at least as precise as the default. The oracle below is sklearn's
/// CONVERGED output (R-CHAR-3), NOT copied from ferrolearn.
/// Tracking: #2131
#[test]
fn divergence_ridge_positive_interior_matches_sklearn() {
    // Symbolic constants from the CONVERGED sklearn 1.5.2 oracle (tol=1e-12),
    // NOT literal-copied from ferrolearn (R-CHAR-3). sklearn's DEFAULT (tol=1e-4)
    // under-converges by ~1.8e-4; these are the true optimum (R-DEV-6).
    const SK_POS_COEF0: f64 = 0.872_119_561_211_034_7;
    const SK_POS_COEF1: f64 = 1.951_998_822_043_437;

    let (x, y) = interior_fixture();

    let positive = Ridge::<f64>::new()
        .with_alpha(0.1)
        .with_positive(true)
        .fit(&x, &y)
        .unwrap();

    // Hard invariant: positive=True => all coef_ >= 0 (this MUST hold).
    for &c in positive.coefficients().iter() {
        assert!(
            c >= 0.0,
            "coef {c} must be non-negative under positive=True"
        );
    }

    // Value parity against the sklearn oracle. sklearn reaches the interior
    // optimum to L-BFGS-B precision; ferrolearn's projected-CD should match it
    // tightly (the constraint is inactive, so this is just the unconstrained
    // optimum that ferrolearn already computes EXACTLY via Cholesky).
    let d0 = (positive.coefficients()[0] - SK_POS_COEF0).abs();
    let d1 = (positive.coefficients()[1] - SK_POS_COEF1).abs();
    assert!(
        d0 <= 1e-6 && d1 <= 1e-6,
        "Ridge(positive=True) interior coef must match sklearn <=1e-6: \
         coef_=[{}, {}] vs sklearn [{SK_POS_COEF0}, {SK_POS_COEF1}] \
         (dev=[{d0:.3e}, {d1:.3e}])",
        positive.coefficients()[0],
        positive.coefficients()[1],
    );
}

/// Internal-consistency parity (no oracle ambiguity): when the positivity
/// constraint is INACTIVE, ferrolearn's `positive=true` fit recovers ferrolearn's
/// OWN exact closed-form unconstrained Cholesky optimum (the inactive-constraint
/// short-circuit returns it exactly). The unconstrained Cholesky path is
/// oracle-verified to match sklearn exactly (REQ-1/5).
///
/// This is NOT tautological: the unconstrained `coefficients()` are the
/// independently oracle-verified closed-form optimum (= sklearn `Ridge.coef_`),
/// used here as the ground-truth target the constrained solve must reach when
/// the constraint binds nowhere.
/// Tracking: #2131
#[test]
fn divergence_ridge_positive_interior_recovers_unconstrained() {
    let (x, y) = interior_fixture();

    let unconstrained = Ridge::<f64>::new().with_alpha(0.1).fit(&x, &y).unwrap();
    // Guard: the unconstrained optimum is genuinely all-positive (interior).
    for &c in unconstrained.coefficients().iter() {
        assert!(
            c > 0.0,
            "fixture must have an interior (all-positive) optimum"
        );
    }

    let positive = Ridge::<f64>::new()
        .with_alpha(0.1)
        .with_positive(true)
        .fit(&x, &y)
        .unwrap();

    for j in 0..2 {
        let dev = (positive.coefficients()[j] - unconstrained.coefficients()[j]).abs();
        assert!(
            dev <= 1e-6,
            "positive=True with an inactive constraint must recover the exact \
             unconstrained optimum (<=1e-6): coef_[{j}]={} vs unconstrained {} \
             (dev={dev:.3e})",
            positive.coefficients()[j],
            unconstrained.coefficients()[j],
        );
    }
}
