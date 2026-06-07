//! Divergence pins for `ferrolearn_linear::ard::ARDRegression` REQ-6 (`scores_`,
//! the per-iteration ARD objective) and REQ-7 (`n_iter_`).
//!
//! sklearn's `ARDRegression.fit` (`sklearn/linear_model/_bayes.py:644-730`)
//! appends ONE objective value per iteration when `compute_score=True`
//! (`_bayes.py:695-704`), INSIDE the loop and BEFORE the convergence break, and
//! never recomputes it afterwards — so `len(scores_) == n_iter_` (unlike
//! `BayesianRidge`, ARD has no post-loop `scores_[-1]` coef-aliasing quirk). The
//! objective (`_bayes.py:696-703`) is
//!
//! ```text
//! s  = sum(lambda_1*log(lambda_) - lambda_2*lambda_)          # over all features
//!    + alpha_1*log(alpha_) - alpha_2*alpha_
//!    + 0.5*(fast_logdet(sigma_) + n_samples*log(alpha_) + sum(log(lambda_)))
//!    - 0.5*(alpha_*rmse_ + sum(lambda_ * coef_**2))
//! ```
//!
//! evaluated with the UPDATED `alpha_`/`lambda_`, the just-PRUNED `coef_`, the
//! PRE-prune `rmse_`, and the kept-block `sigma_` from the TOP of the iteration.
//! `n_iter_ = iter_ + 1` (`_bayes.py:716`).
//!
//! Expected values are from the LIVE sklearn 1.5.2 oracle (NOT copied from
//! ferrolearn — R-CHAR-3):
//!
//! ```text
//! python3 -c "import numpy as np; from sklearn.linear_model import ARDRegression; \
//!   X=np.array([[1.,100.],[2.,200.],[3.,300.],[4.,400.],[5.,500.],[6.,600.]]); \
//!   y=np.array([2.,4.,6.,8.,10.,12.]); \
//!   m=ARDRegression(max_iter=1000, compute_score=True).fit(X,y); \
//!   print(m.n_iter_, m.scores_.tolist())"
//! # -> 3 [41.81941902228245, 31.770026461186802, 31.695712514997574]
//! ```
//!
//! Tracking: #477 (REQ-6 scores_), #478 (REQ-7 n_iter_).

use ferrolearn_core::traits::Fit;
use ferrolearn_linear::ard::ARDRegression;
use ndarray::{Array2, array};

/// REQ-6/7 (default hyperpriors, 2-feature design): `scores_` matches the live
/// sklearn 1.5.2 per-iteration objective sequence and `n_iter_ == 3`.
#[test]
fn divergence_ard_scores_2feature_default() {
    // sklearn 1.5.2 live oracle (see module doc).
    const SK_N_ITER: usize = 3;
    const SK_SCORES: [f64; 3] = [
        41.819_419_022_282_45,
        31.770_026_461_186_802,
        31.695_712_514_997_574,
    ];

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
        .with_compute_score(true)
        .fit(&x, &y)
        .unwrap();

    assert_eq!(fitted.n_iter(), SK_N_ITER, "n_iter_ diverges (#478)");
    let scores = fitted.scores();
    assert_eq!(
        scores.len(),
        SK_N_ITER,
        "scores_ length must equal n_iter_ (no post-loop append) (#477): got {scores:?}",
    );
    for (i, (&got, &want)) in scores.iter().zip(SK_SCORES.iter()).enumerate() {
        // Per the BayesianRidge audit, when the EM iterates align the per-iter
        // LML matches to ~1e-12.
        assert!(
            (got - want).abs() <= 1e-9 * want.abs().max(1.0),
            "scores_[{i}] diverges: ferrolearn={got}, sklearn={want} (#477)",
        );
    }
}

/// REQ-6/7 (NON-default hyperpriors, 2-feature design): the full oscillating
/// objective sequence + `n_iter_ == 6` match the live oracle.
#[test]
fn divergence_ard_scores_2feature_nondefault_hyperpriors() {
    // sklearn 1.5.2 live oracle:
    //   ARDRegression(max_iter=1000, compute_score=True, alpha_1=1e-3,
    //                 alpha_2=1e-3, lambda_1=1e-3, lambda_2=1e-3)
    const SK_N_ITER: usize = 6;
    const SK_SCORES: [f64; 6] = [
        18.717_300_469_769_185,
        14.314_722_469_591_18,
        13.470_263_797_311_89,
        13.594_820_341_093_962,
        13.521_065_079_837_788,
        13.561_286_359_592_392,
    ];

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
        .with_compute_score(true)
        .with_alpha_1(1e-3)
        .with_alpha_2(1e-3)
        .with_lambda_1(1e-3)
        .with_lambda_2(1e-3)
        .fit(&x, &y)
        .unwrap();

    assert_eq!(fitted.n_iter(), SK_N_ITER, "n_iter_ diverges (#478)");
    let scores = fitted.scores();
    assert_eq!(
        scores.len(),
        SK_N_ITER,
        "scores_ length must equal n_iter_ (#477)"
    );
    for (i, (&got, &want)) in scores.iter().zip(SK_SCORES.iter()).enumerate() {
        assert!(
            (got - want).abs() <= 1e-9 * want.abs().max(1.0),
            "scores_[{i}] diverges: ferrolearn={got}, sklearn={want} (#477)",
        );
    }
}

/// REQ-6/7 (4-feature mixed-relevance design, `y = 2*x0 + 0.5*x2`): the
/// objective sequence with PRUNED features (1, 3) + `n_iter_ == 4` match the
/// live oracle. This exercises the `fast_logdet(sigma_)` term on a kept-block
/// `sigma_` smaller than `n_features`.
#[test]
fn divergence_ard_scores_4feature_pruned() {
    // sklearn 1.5.2 live oracle.
    const SK_N_ITER: usize = 4;
    const SK_SCORES: [f64; 4] = [
        -6.481_207_066_864_622,
        19.576_534_162_589_45,
        60.992_432_068_241_75,
        56.262_296_370_526_755,
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
        .with_compute_score(true)
        .fit(&x, &y)
        .unwrap();

    assert_eq!(fitted.n_iter(), SK_N_ITER, "n_iter_ diverges (#478)");
    let scores = fitted.scores();
    assert_eq!(
        scores.len(),
        SK_N_ITER,
        "scores_ length must equal n_iter_ (#477)"
    );
    for (i, (&got, &want)) in scores.iter().zip(SK_SCORES.iter()).enumerate() {
        assert!(
            (got - want).abs() <= 1e-8 * want.abs().max(1.0),
            "scores_[{i}] diverges: ferrolearn={got}, sklearn={want} (#477)",
        );
    }
}

/// REQ-6: `compute_score = false` (the default) leaves `scores_` empty, matching
/// sklearn (which only populates `scores_` under `compute_score=True`).
#[test]
fn ard_scores_empty_without_compute_score() {
    let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

    let fitted = ARDRegression::<f64>::new().fit(&x, &y).unwrap();
    assert!(
        fitted.scores().is_empty(),
        "scores_ must be empty when compute_score=false",
    );
    // n_iter_ is still populated regardless of compute_score (#478).
    assert!(fitted.n_iter() >= 1);
}
