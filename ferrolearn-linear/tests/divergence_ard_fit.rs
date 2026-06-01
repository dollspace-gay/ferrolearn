//! Divergence pin for `ferrolearn_linear::ard::ARDRegression::fit`.
//!
//! sklearn's `ARDRegression.fit` (`sklearn/linear_model/_bayes.py:644-730`)
//! drives a PER-ITERATION `keep_lambda = lambda_ < threshold_lambda` mask
//! (`_bayes.py:691`: `keep_lambda = lambda_ < self.threshold_lambda`) so that
//! only KEPT columns are re-solved each iteration, and seeds
//! `alpha_ = 1.0 / (np.var(y) + eps)` (`_bayes.py:658`:
//! `alpha_ = np.asarray(1.0 / (np.var(y) + eps), dtype=dtype)`), converging on
//! `sum(|coef_old - coef_|) < tol` (`_bayes.py:707`).
//!
//! ferrolearn's `fn fit` (`ard.rs`) instead solves the FULL feature system every
//! iteration (`fn ard_solve` over all `n_features`, no `keep_lambda`), seeds
//! `let mut alpha = F::one()` (not `1/Var(y)`), and prunes ONCE after the loop
//! (`if lambda[i] > self.threshold_lambda { w[i] = F::zero() }`). The result is
//! a DIFFERENT pruning trajectory: features sklearn keeps are wrongly zeroed.
//!
//! Expected values are from the LIVE sklearn 1.5.2 oracle (NOT copied from
//! ferrolearn — R-CHAR-3):
//!
//! ```text
//! # 2-feature design
//! python3 -c "from sklearn.linear_model import ARDRegression; import numpy as np; \
//!   X=np.array([[1.,100.],[2.,200.],[3.,300.],[4.,400.],[5.,500.],[6.,600.]]); \
//!   y=np.array([2.,4.,6.,8.,10.,12.]); m=ARDRegression(max_iter=1000).fit(X,y); \
//!   print(m.coef_.tolist(), float(m.alpha_), m.lambda_.tolist())"
//! # -> [0.010193278903540725, 0.019898067101591296] 2500000.999899827 \
//! #    [48.80203149344805, 2500.0051052035033]
//!
//! # 4-feature design (y = 2*x0 + 0.5*x2; features 1,3 irrelevant)
//! python3 -c "from sklearn.linear_model import ARDRegression; import numpy as np; \
//!   X=np.array([[1.,50.,2.,-3.],[2.,10.,4.,1.],[3.,90.,6.,-7.],[4.,20.,8.,5.], \
//!               [5.,70.,10.,-2.],[6.,40.,12.,9.],[7.,60.,14.,-1.],[8.,30.,16.,4.]]); \
//!   y=2.0*X[:,0]+0.5*X[:,2]; m=ARDRegression(max_iter=1000).fit(X,y); \
//!   print(m.coef_.tolist(), float(m.alpha_), m.lambda_.tolist())"
//! # -> [0.5999942561395434, 0.0, 1.2000028896712696, 0.0] 3500000.977... \
//! #    [0.5555..., 178801.86..., 0.5555..., 10221.71...]
//! ```
//!
//! Tracking: #474 (Blocker for REQ-1 of ard: iterative ARD fit with per-iter
//! keep_lambda masking + `1/Var(y)` init + coef-delta convergence).

use ferrolearn_core::introspection::HasCoefficients;
use ferrolearn_core::traits::Fit;
use ferrolearn_linear::ard::ARDRegression;
use ndarray::{Array2, array};

/// Set of indices `i` where `coef_[i] == 0.0` (the pruned/relevance-determined set).
fn pruned_set(coef: &ndarray::Array1<f64>) -> Vec<usize> {
    coef.iter()
        .enumerate()
        .filter_map(|(i, &c)| if c == 0.0 { Some(i) } else { None })
        .collect()
}

/// Divergence: `ARDRegression::fit` on the doc-author's 2-feature design.
///
/// sklearn (`_bayes.py:644-730`) keeps BOTH features:
///   coef_   = [0.010193278903540725, 0.019898067101591296]  (pruned set = {})
///   alpha_  = 2500000.999899827
///   lambda_ = [48.80203149344805, 2500.0051052035033]
/// ferrolearn (no per-iter `keep_lambda` mask + `alpha=F::one()` init) WRONGLY
/// prunes feature 0:
///   coef    = [0.0, 0.019999989818706716]                   (pruned set = {0})
///   alpha   = 2009390.2500119198
///   lambda  = [493087.1732538648, 2475.252879252219]
///
/// Tracking: #474
#[test]
#[ignore = "divergence: ARD fit has no per-iter keep_lambda mask + wrong init (alpha=1, not 1/Var(y)); feature 0 wrongly pruned on multi-feature designs; tracking #474"]
fn divergence_ard_fit_2feature_wrong_pruning() {
    // sklearn 1.5.2 live oracle (see module doc).
    const SK_COEF: [f64; 2] = [0.010193278903540725, 0.019898067101591296];
    const SK_ALPHA: f64 = 2500000.999899827;
    const SK_LAMBDA: [f64; 2] = [48.80203149344805, 2500.0051052035033];
    const SK_PRUNED: [usize; 0] = []; // sklearn keeps both features.

    let x = Array2::from_shape_vec(
        (6, 2),
        vec![
            1.0, 100.0, 2.0, 200.0, 3.0, 300.0, 4.0, 400.0, 5.0, 500.0, 6.0, 600.0,
        ],
    )
    .unwrap();
    let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0];

    let fitted = ARDRegression::<f64>::new()
        .with_max_iter(1000)
        .fit(&x, &y)
        .unwrap();

    let coef = fitted.coefficients();

    // (1) Pruned (relevance) set must match sklearn: sklearn keeps both; ferrolearn drops feature 0.
    assert_eq!(
        pruned_set(coef),
        SK_PRUNED.to_vec(),
        "pruned set diverges: ferrolearn coef={:?} prunes {:?}, sklearn keeps both (#474)",
        coef.to_vec(),
        pruned_set(coef),
    );

    // (2) coef_ must match sklearn element-wise.
    for i in 0..2 {
        let diff = (coef[i] - SK_COEF[i]).abs();
        let tol = 1e-6 * SK_COEF[i].abs().max(1e-6);
        assert!(
            diff <= tol,
            "coef_[{i}] diverges: ferrolearn={}, sklearn={} (#474)",
            coef[i],
            SK_COEF[i],
        );
    }

    // (3) alpha_ (noise precision) must match sklearn.
    {
        let diff = (fitted.alpha() - SK_ALPHA).abs();
        assert!(
            diff <= 1e-3 * SK_ALPHA.abs(),
            "alpha_ diverges: ferrolearn={}, sklearn={} (#474)",
            fitted.alpha(),
            SK_ALPHA,
        );
    }

    // (4) lambda_ (per-feature weight precision) must match sklearn.
    let lambda = fitted.lambda();
    for i in 0..2 {
        let diff = (lambda[i] - SK_LAMBDA[i]).abs();
        assert!(
            diff <= 1e-3 * SK_LAMBDA[i].abs(),
            "lambda_[{i}] diverges: ferrolearn={}, sklearn={} (#474)",
            lambda[i],
            SK_LAMBDA[i],
        );
    }
}

/// Divergence: `ARDRegression::fit` on a 4-feature mixed-relevance design
/// (`y = 2*x0 + 0.5*x2`; features 1 and 3 are irrelevant).
///
/// sklearn (`_bayes.py:644-730`) prunes ONLY the genuinely-irrelevant features:
///   coef_       = [0.5999942561395434, 0.0, 1.2000028896712696, 0.0]
///   pruned set  = {1, 3}
///   alpha_      = 3500000.9771208703
///   lambda_     = [0.5555620744..., 178801.86499..., 0.5555551871..., 10221.713089...]
/// ferrolearn (full-system solve + post-loop pruning + `alpha=1` init) WRONGLY
/// prunes feature 0 as well:
///   coef        = [0.0, 0.0, 1.4999996612387911, 0.0]
///   pruned set  = {0, 1, 3}
///   alpha       = 2005885.9824054108
///
/// Tracking: #474
#[test]
#[ignore = "divergence: ARD fit prunes a relevant feature (0) it should keep; no per-iter keep_lambda mask + wrong init; tracking #474"]
fn divergence_ard_fit_4feature_wrong_pruned_set() {
    // sklearn 1.5.2 live oracle (see module doc).
    const SK_COEF: [f64; 4] = [0.5999942561395434, 0.0, 1.2000028896712696, 0.0];
    const SK_PRUNED: [usize; 2] = [1, 3]; // sklearn keeps relevant features 0 and 2.
    const SK_ALPHA: f64 = 3500000.9771208703;
    const SK_LAMBDA: [f64; 4] = [
        0.5555620744276873,
        178801.86499302133,
        0.5555551871724366,
        10221.713089166677,
    ];

    let x = Array2::from_shape_vec(
        (8, 4),
        vec![
            1.0, 50.0, 2.0, -3.0, 2.0, 10.0, 4.0, 1.0, 3.0, 90.0, 6.0, -7.0, 4.0, 20.0, 8.0, 5.0,
            5.0, 70.0, 10.0, -2.0, 6.0, 40.0, 12.0, 9.0, 7.0, 60.0, 14.0, -1.0, 8.0, 30.0, 16.0,
            4.0,
        ],
    )
    .unwrap();
    let y = array![3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0];

    let fitted = ARDRegression::<f64>::new()
        .with_max_iter(1000)
        .fit(&x, &y)
        .unwrap();

    let coef = fitted.coefficients();

    // (1) Pruned set must be exactly {1, 3}; ferrolearn additionally prunes 0.
    assert_eq!(
        pruned_set(coef),
        SK_PRUNED.to_vec(),
        "pruned set diverges: ferrolearn prunes {:?} (coef={:?}), sklearn prunes {:?} (#474)",
        pruned_set(coef),
        coef.to_vec(),
        SK_PRUNED,
    );

    // (2) coef_ must match sklearn element-wise (kept features carry nonzero values).
    for i in 0..4 {
        let diff = (coef[i] - SK_COEF[i]).abs();
        let tol = 1e-5 * SK_COEF[i].abs().max(1e-6);
        assert!(
            diff <= tol,
            "coef_[{i}] diverges: ferrolearn={}, sklearn={} (#474)",
            coef[i],
            SK_COEF[i],
        );
    }

    // (3) alpha_ must match sklearn.
    {
        let diff = (fitted.alpha() - SK_ALPHA).abs();
        assert!(
            diff <= 1e-3 * SK_ALPHA.abs(),
            "alpha_ diverges: ferrolearn={}, sklearn={} (#474)",
            fitted.alpha(),
            SK_ALPHA,
        );
    }

    // (4) lambda_ on the KEPT features (0, 2) must match sklearn; pruned features
    // (1, 3) drive lambda toward the threshold either way, so only assert kept ones.
    let lambda = fitted.lambda();
    for &i in &[0usize, 2usize] {
        let diff = (lambda[i] - SK_LAMBDA[i]).abs();
        assert!(
            diff <= 1e-3 * SK_LAMBDA[i].abs().max(1e-6),
            "lambda_[{i}] diverges: ferrolearn={}, sklearn={} (#474)",
            lambda[i],
            SK_LAMBDA[i],
        );
    }
}

/// CONTROL (sanity, NOT a divergence): on a SINGLE-feature design the
/// `keep_lambda` mask is trivially a no-op, so ferrolearn matches sklearn.
/// This documents the regime where the #474 bug does NOT manifest. It is left
/// UN-ignored and is expected to PASS — if it ever fails, the fix for #474
/// regressed the 1D path.
///
/// sklearn 1.5.2 live oracle:
///   X=[[1],[2],[3],[4],[5]], y=[3,5,7,9,11]
///   -> coef_=[1.9999999724...], intercept_=1.0000000827..., alpha_=2.000001e6,
///      lambda_=[0.2500003...], n_iter_=5
#[test]
fn ard_fit_1feature_matches_sklearn_control() {
    // sklearn 1.5.2 live oracle.
    const SK_COEF0: f64 = 1.9999999724;
    const SK_INTERCEPT: f64 = 1.0000000827;
    const SK_ALPHA: f64 = 2.000001e6;

    let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

    let fitted = ARDRegression::<f64>::new().fit(&x, &y).unwrap();

    assert!(
        (fitted.coefficients()[0] - SK_COEF0).abs() <= 1e-4,
        "1D coef_ should match sklearn: ferrolearn={}, sklearn={}",
        fitted.coefficients()[0],
        SK_COEF0,
    );
    assert!(
        (fitted.intercept() - SK_INTERCEPT).abs() <= 1e-3,
        "1D intercept_ should match sklearn: ferrolearn={}, sklearn={}",
        fitted.intercept(),
        SK_INTERCEPT,
    );
    assert!(
        (fitted.alpha() - SK_ALPHA).abs() <= 1e-2 * SK_ALPHA,
        "1D alpha_ should match sklearn: ferrolearn={}, sklearn={}",
        fitted.alpha(),
        SK_ALPHA,
    );
}
