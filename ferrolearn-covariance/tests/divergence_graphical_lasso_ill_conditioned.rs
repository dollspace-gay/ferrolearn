//! Divergence test pinning `ferrolearn-covariance` `GraphicalLasso` against the
//! live scikit-learn 1.5.2 oracle on an ILL-CONDITIONED empirical covariance.
//!
//! Authored adversarially (ACToR critic role). The expected `precision_` below
//! was computed by running scikit-learn 1.5.2 on the EXACT `X` embedded in
//! [`ill_conditioned_x`] and copied from the sklearn side ONLY (R-CHAR-3 — never
//! from ferrolearn). Reproduce with:
//!
//! ```text
//! import numpy as np
//! from sklearn.covariance import GraphicalLasso
//! rng = np.random.RandomState(7); n, p = 6, 5
//! X = rng.randn(n, p)
//! X[:, 1] = X[:, 0] + 1e-6 * rng.randn(n)   # feature 1 near-collinear w/ feature 0
//! m = GraphicalLasso(alpha=0.01, max_iter=100, tol=1e-4).fit(X)
//! m.precision_   # m.n_iter_ == 8
//! ```
//!
//! `empirical_covariance(X)` has condition number ~9e12 (smallest eigenvalue
//! ~4e-13), so the two near-collinear features make `emp_cov` near-singular.
//! sklearn produces a finite result (no ConvergenceWarning) in `n_iter_ = 8`.

use ferrolearn_core::traits::Fit;
use ferrolearn_covariance::GraphicalLasso;
use ndarray::{Array2, array};

/// Near-singular probe: `RandomState(7).randn(6, 5)` with feature 1 overwritten
/// by `feature0 + 1e-6 * randn(6)`. Python `repr` (shortest round-trip) literals.
fn ill_conditioned_x() -> Array2<f64> {
    array![
        [
            1.690525703800356,
            1.690524961246831,
            0.0328201636785844,
            0.40751628299650783,
            -0.7889230286257386
        ],
        [
            0.00206557290594813,
            0.0020666453760797058,
            -1.7547243063454208,
            1.0176580056634932,
            0.6004985159195494
        ],
        [
            -0.6254289739667597,
            -0.625430625042349,
            0.5052993741967516,
            -0.261356415191647,
            -0.2427490786725466
        ],
        [
            -1.4532414124907906,
            -1.4532408770614342,
            0.12388090528703843,
            0.2744599237599636,
            -1.5265245318698402
        ],
        [
            1.6506996911864755,
            1.6506976267716724,
            -0.3871399432863881,
            2.029072220761112,
            -0.04538602986064609
        ],
        [
            -1.4506786991465748,
            -1.4506793613059143,
            -2.2883151019717225,
            1.0493965493432547,
            -0.41647431852001854
        ],
    ]
}

/// sklearn 1.5.2 `GraphicalLasso(alpha=0.01).fit(ill_conditioned_x()).precision_`
/// (row-major). Computed live; copied from the sklearn side only (R-CHAR-3).
fn sklearn_precision() -> Array2<f64> {
    let flat = [
        50.30372875080707,
        -49.681279496488536,
        -0.5516896849603299,
        -0.6736316194145492,
        -0.4129316282682264,
        -49.681279496488536,
        50.30318012265854,
        -0.5509240238128769,
        -0.6731497815660947,
        -0.41234156001741995,
        -0.5516896849603299,
        -0.5509240238128769,
        2.3391558187329653,
        1.8529464866933312,
        1.3972695647234359,
        -0.6736316194145492,
        -0.6731497815660947,
        1.8529464866933312,
        4.157845628918763,
        0.02497927913640859,
        -0.4129316282682264,
        -0.41234156001741995,
        1.3972695647234359,
        0.02497927913640859,
        3.801312488014349,
    ];
    Array2::from_shape_vec((5, 5), flat.to_vec()).unwrap()
}

fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

/// Divergence: ferrolearn's inner Gram coordinate descent
/// ([`enet_coordinate_descent_gram`], `graphical_lasso.rs:615-617`) breaks
/// directly on the max-change rule `d_w_max / w_max < enet_tol`, whereas
/// scikit-learn's `enet_coordinate_descent_gram`
/// (`sklearn/linear_model/_cd_fast.pyx:685-727`) uses that rule ONLY as a gate
/// to then compute a full duality gap and break on
/// `gap < tol` where `tol` was rescaled by `y_norm2 = np.dot(y, y)`
/// (`:635 tol = tol * y_norm2`; `:725 if gap < tol: break`). ferrolearn never
/// computes the duality gap and never rescales `enet_tol` by `y_norm2`.
///
/// On well-conditioned inputs the two stopping rules coincide to ~1e-16 (the
/// 4x4 probe over alpha in {0.01,0.05,0.1,0.2,0.3,0.5}, and a p=6/n=80 system,
/// all match — including `n_iter`). But on the near-singular `emp_cov` above
/// (condition number ~9e12) at `alpha=0.01` the inner CD terminates at a
/// different point:
///
///   sklearn precision_[0,0] = 50.30372875080707   (n_iter_ = 8)
///   ferrolearn precision()[0,0] = 50.30747905772279 (n_iter()  = 9)
///
/// precision_ max-abs-diff = ~3.9e-3 (>> the 1e-4 convergence contract).
/// Swapping ONLY the inner-CD stop in a pure-python port reproduces ferrolearn's
/// output to 5.6e-13, confirming the inner criterion as the sole root cause.
///
/// Tracking: #1889 (new-blocker).
#[test]
fn divergence_graphical_lasso_ill_conditioned_inner_cd_alpha_0_01() {
    // #1889 new-blocker
    let x = ill_conditioned_x();
    let fitted = GraphicalLasso::<f64>::new(0.01)
        .max_iter(100)
        .tol(1e-4)
        .fit(&x, &())
        .unwrap();

    let exp_prec = sklearn_precision();
    let prec_diff = max_abs_diff(fitted.precision(), &exp_prec);

    // sklearn's documented convergence tolerance is 1e-4; pin to that contract.
    assert!(
        prec_diff < 1e-4,
        "precision_ diverges from sklearn on ill-conditioned emp_cov: \
         max-abs-diff = {prec_diff:e} (>= 1e-4)\n\
         ferrolearn precision()[0,0] = {} (n_iter = {}); sklearn precision_[0,0] = {}",
        fitted.precision()[[0, 0]],
        fitted.n_iter(),
        exp_prec[[0, 0]],
    );
}
