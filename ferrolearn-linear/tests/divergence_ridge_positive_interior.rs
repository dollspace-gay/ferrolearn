//! Adversarial divergence audit of `ferrolearn_linear::Ridge` with
//! `positive=true` at an INTERIOR optimum, against the live scikit-learn 1.5.2
//! oracle.
//!
//! Context: the `positive=true` path was surfaced through the PyO3 binding +
//! Python wrappers (#2129). Active-constraint fixtures (a coefficient clamped to
//! 0) match sklearn to machine epsilon, because the binding correctly threads
//! `positive` and the vertex solution is uniquely determined. This file pins the
//! REMAINING divergence: at an INTERIOR optimum (the unconstrained ridge solution
//! is already all-positive, so `positive=true` is a NO-OP constraint), the
//! downstream projected coordinate-descent solver
//! (`ferrolearn-linear/src/ridge.rs:295` `with_positive` -> `solve_nonneg_ridge`,
//! REQ-9 #387) stops at a different iterate than sklearn's L-BFGS-B
//! (`sklearn/linear_model/_ridge.py:923-928` dispatch -> `_solve_lbfgs`
//! `_ridge.py:300`), AND fails to recover ferrolearn's OWN exact closed-form
//! unconstrained Cholesky optimum.
//!
//! All expected values are computed by live-calling sklearn 1.5.2 (R-CHAR-3); no
//! value is literal-copied from the ferrolearn side.
//!
//! Tracking: #2131 (downstream solver gap, #387/#412 lineage). The #2129 BINDING
//! SURFACE is correct; this is a separate downstream precision gap.

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
        vec![
            1.0, 0.2, 0.3, 1.0, 0.5, 0.5, 1.2, 0.1, 0.1, 1.3, 0.8, 0.9,
        ],
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

/// Divergence: `ferrolearn_linear::Ridge(alpha=0.1, positive=true)` diverges from
/// `sklearn/linear_model/_ridge.py:923-928` for an INTERIOR optimum.
///
/// Live sklearn 1.5.2 oracle:
/// ```text
/// python3 -c "import numpy as np; from sklearn.linear_model import Ridge; \
///   X=np.array([[1.,0.2],[0.3,1.],[0.5,0.5],[1.2,0.1],[0.1,1.3],[0.8,0.9]]); \
///   y=X@np.array([2.,3.]); \
///   print(Ridge(alpha=0.1,positive=True).fit(X,y).coef_.tolist())"
/// # -> [0.8719358921480487, 1.9518078349228183]
/// # unconstrained Ridge(alpha=0.1).coef_ -> [0.87211956, 1.95199882] (all > 0)
/// ```
/// ferrolearn returns `[0.87198097, 1.95189993]` (projected-CD), deviating
/// ~9.2e-5 from sklearn's L-BFGS-B AND ~1.4e-4 from ferrolearn's OWN exact
/// closed-form unconstrained Cholesky optimum.
/// Tracking: #2131
#[test]
#[ignore = "divergence: Ridge(positive=True) projected-CD vs sklearn L-BFGS-B interior-optimum gap ~9e-5; tracking #2131"]
fn divergence_ridge_positive_interior_matches_sklearn() {
    // Symbolic constants traceable to the sklearn 1.5.2 oracle (R-CHAR-3).
    const SK_POS_COEF0: f64 = 0.871_935_892_148_048_7;
    const SK_POS_COEF1: f64 = 1.951_807_834_922_818_3;

    let (x, y) = interior_fixture();

    let positive = Ridge::<f64>::new()
        .with_alpha(0.1)
        .with_positive(true)
        .fit(&x, &y)
        .unwrap();

    // Hard invariant: positive=True => all coef_ >= 0 (this MUST hold).
    for &c in positive.coefficients().iter() {
        assert!(c >= 0.0, "coef {c} must be non-negative under positive=True");
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

/// Internal-consistency divergence (no oracle ambiguity): when the positivity
/// constraint is INACTIVE, ferrolearn's projected-CD `positive=true` fit must
/// recover ferrolearn's OWN exact closed-form unconstrained Cholesky optimum.
/// It does not — it stops ~1.4e-4 away. The unconstrained Cholesky path is
/// oracle-verified to match sklearn exactly (REQ-1/5), so this gap is purely the
/// positive solver's stopping criterion.
///
/// This is NOT tautological: the unconstrained `coefficients()` are the
/// independently oracle-verified closed-form optimum (= sklearn `Ridge.coef_`),
/// used here as the ground-truth target the constrained solve should reach when
/// the constraint binds nowhere.
/// Tracking: #2131
#[test]
#[ignore = "divergence: Ridge(positive=True) interior solve fails to recover its own exact unconstrained optimum (~1.4e-4); tracking #2131"]
fn divergence_ridge_positive_interior_recovers_unconstrained() {
    let (x, y) = interior_fixture();

    let unconstrained = Ridge::<f64>::new().with_alpha(0.1).fit(&x, &y).unwrap();
    // Guard: the unconstrained optimum is genuinely all-positive (interior).
    for &c in unconstrained.coefficients().iter() {
        assert!(c > 0.0, "fixture must have an interior (all-positive) optimum");
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
