//! Divergence pins for `RANSACRegressor::fit` against the live scikit-learn
//! 1.5.2 oracle (`sklearn/linear_model/_ransac.py`, commit 156ef14).
//!
//! # Why these tests avoid RNG-sequence parity
//!
//! sklearn draws candidate subsets with `sample_without_replacement` over
//! numpy's Mersenne-Twister (`sklearn/linear_model/_ransac.py:478`); ferrolearn
//! draws with a Fisher-Yates partial shuffle over `rand::rngs::StdRng`
//! (`ferrolearn-linear/src/ransac.rs:173-181`). The two PRNGs produce different
//! subset *sequences* from any seed, so the exact subset trajectory is NOT
//! comparable. Each test below uses a dataset whose consensus-set decision is
//! determined by the *rule*, with the candidate space small enough and
//! `max_trials` large enough that both implementations exhaust it — so a
//! divergence in the REPORTED model is attributable to the decision rule, not
//! to which subset happened to be drawn. Both pins were re-verified robust
//! across sklearn `random_state` sweeps (D1: 0..=11, D5: 0..=29).
//!
//! Every expected value is produced by RUNNING scikit-learn 1.5.2 (the live
//! oracle), never copied from ferrolearn (goal.md R-CHAR-3). The exact python
//! invocation is recorded above each block of oracle constants.
//!
//! # Divergence that could NOT be RNG-pinned (D2 / REQ-5)
//!
//! sklearn records `inlier_mask_` from the winning SUBSET model and refits on
//! its inliers exactly ONCE after the loop, never recomputing the mask
//! (`_ransac.py:544,602,605`); ferrolearn refits INSIDE the loop and stores the
//! mask recomputed from the refitted model (`ransac.rs:312-324`). This is a
//! real divergence, but with a `LinearRegression` base the winning consensus
//! set is genuinely coupled to the (non-parityable) subset draw: datasets where
//! the subset mask and the refit-recompute mask differ also have an RNG-coupled
//! winner (sklearn itself returns 2+ distinct masks across large seed sweeps,
//! e.g. `{[1,1,0,1,1,0], [1,0,1,0,0,1]}` on the `y≈x` scatter probed for #513),
//! and ferrolearn's independent draw lands on its own winner — so no single
//! mask/count assertion fails deterministically against the current code. D2 is
//! reported as a documented un-pinnable-at-this-layer divergence; pinning it
//! deterministically requires either a base estimator whose subset fit is
//! draw-invariant or a refit-count/refit-site observable (e.g. a spy estimator
//! counting `fit` calls), neither of which the current public API exposes.

use ferrolearn_core::traits::{Fit, Predict};
use ferrolearn_linear::LinearRegression;
use ferrolearn_linear::ransac::RANSACRegressor;
use ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// D1 / REQ-4 — selection criterion: sklearn ranks tied consensus sets by the
// base estimator's R^2 score (HIGHER wins); ferrolearn ranks by residual_sum
// (LOWER wins). These disagree on the dataset below.
// ---------------------------------------------------------------------------

/// Divergence: `RANSACRegressor::fit` selects the wrong consensus set on a tie.
///
/// sklearn site: `sklearn/linear_model/_ransac.py:530-543`
///   `score_subset = estimator.score(X_inlier_subset, y_inlier_subset, ...)`
///   `if n_inliers_subset == n_inliers_best and score_subset < score_best: continue`
///   `score_best = score_subset` — i.e. on a tie in inlier count the HIGHER R^2
///   consensus set wins.
///
/// ferrolearn site: `ferrolearn-linear/src/ransac.rs:298-299`
///   `let is_better = n_inliers > best_n_inliers
///        || (n_inliers == best_n_inliers && residual_sum < best_residual_sum);`
///   — on a tie the LOWER absolute-residual-sum consensus set wins.
///
/// Dataset (two collinear groups sharing x in {0,1,2}, separated in y) — with
/// `residual_threshold = 0.3` there are exactly two 3-inlier consensus sets,
/// verified by enumerating all `C(6,2)` two-point subsets:
///
/// ```text
/// X = [[0],[1],[2],[0],[1],[2]]
/// y = [5.0, 5.02, 4.98,  0.0, 10.15, 20.0]
/// group A = idx {0,1,2}: best subset R^2 = -0.125, residual_sum = 0.03
/// group B = idx {3,4,5}: best subset R^2 =  0.9999, residual_sum = 0.15
/// ```
///
/// sklearn (higher R^2) picks B; ferrolearn (lower residual_sum) picks A.
///
/// Oracle (sklearn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; from sklearn.linear_model import \
///   RANSACRegressor, LinearRegression; \
///   X=np.array([[0.],[1.],[2.],[0.],[1.],[2.]]); \
///   y=np.array([5.0,5.02,4.98,0.0,10.15,20.0]); \
///   m=RANSACRegressor(LinearRegression(), random_state=0, \
///     residual_threshold=0.3, max_trials=1000).fit(X,y); \
///   print(m.inlier_mask_.astype(int).tolist(), \
///     round(m.predict([[1.0]])[0],6), round(m.estimator_.coef_[0],6), \
///     round(m.estimator_.intercept_,6))"
/// # -> [0, 0, 0, 1, 1, 1] 10.05 10.0 0.05
/// ```
/// (robust: identical mask across random_state 0..=11.)
///
/// Tracking: #512
#[test]
fn ransac_selection_criterion_r2_not_residual_sum() {
    let x = Array2::from_shape_vec((6, 1), vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0]).unwrap();
    let y = Array1::from(vec![5.0, 5.02, 4.98, 0.0, 10.15, 20.0]);

    let model = RANSACRegressor::new(LinearRegression::<f64>::new())
        .with_residual_threshold(0.3)
        .with_max_trials(1000)
        .with_random_state(0);
    let fitted = model.fit(&x, &y).expect("RANSAC fit");

    // sklearn picks group B (high R^2): inlier_mask_ == [F,F,F,T,T,T].
    const SK_MASK: [bool; 6] = [false, false, false, true, true, true];
    let mask = fitted.inlier_mask();
    assert_eq!(
        mask, &SK_MASK,
        "selection criterion: sklearn (higher R^2 wins ties) selects group B \
         mask {SK_MASK:?}; ferrolearn (lower residual_sum) reports {mask:?}"
    );

    // And the refit-on-winner estimator predicts the B-line at x=1 (~10.05),
    // not the A-line (~5.0). Oracle: m.predict([[1.0]]) == 10.05.
    const SK_PRED_AT_1: f64 = 10.05;
    let pred = fitted
        .predict(&Array2::from_shape_vec((1, 1), vec![1.0]).unwrap())
        .expect("predict");
    assert!(
        (pred[0] - SK_PRED_AT_1).abs() < 1e-6,
        "winner refit: sklearn predicts {SK_PRED_AT_1} at x=1 (group-B line); \
         ferrolearn predicts {} (group-A line)",
        pred[0]
    );
}

// ---------------------------------------------------------------------------
// D5 / REQ-9 — MAD-zero threshold: sklearn uses residual_threshold = MAD(y),
// which is exactly 0 for a near-constant target; ferrolearn substitutes 1e-6.
// ---------------------------------------------------------------------------

/// Divergence: `RANSACRegressor::fit` uses the wrong auto threshold when MAD = 0.
///
/// sklearn site: `sklearn/linear_model/_ransac.py:399-401`
///   `if self.residual_threshold is None:`
///   `    residual_threshold = np.median(np.abs(y - np.median(y)))`
///   — no special-casing; the threshold can be exactly 0.0.
///
/// ferrolearn site: `ferrolearn-linear/src/ransac.rs:249-254`
///   `if y_mad <= F::epsilon() { F::from(1e-6).unwrap() } else { y_mad }`
///   — substitutes 1e-6 when MAD is ~0.
///
/// Dataset: 7 points exactly on y = 5 plus one tiny deviation in `(0, 1e-6)`:
///
/// ```text
/// X = [[0]..[7]]
/// y = [5,5,5,5,5,5,5, 5.0000001]
/// ```
///
/// MAD(y) = median(|y - median(y)|) = 0.0 (oracle below). With threshold 0,
/// sklearn classifies idx 7 (residual 1e-7) as an OUTLIER. ferrolearn's 1e-6
/// threshold (>= 1e-7) classifies idx 7 as an INLIER. The masks differ at idx 7.
///
/// Oracle (sklearn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; from sklearn.linear_model import \
///   RANSACRegressor, LinearRegression; \
///   X=np.arange(8).reshape(-1,1).astype(float); \
///   y=np.array([5.,5.,5.,5.,5.,5.,5.,5.0000001]); \
///   print(np.median(np.abs(y-np.median(y)))); \
///   m=RANSACRegressor(LinearRegression(), random_state=0, \
///     max_trials=200).fit(X,y); print(m.inlier_mask_.astype(int).tolist())"
/// # -> 0.0
/// # -> [1, 1, 1, 1, 1, 1, 1, 0]
/// ```
/// (robust: identical mask across random_state 0..=29.)
///
/// Tracking: #517
#[test]
fn ransac_mad_zero_threshold_excludes_tiny_deviation() {
    let x = Array2::from_shape_vec((8, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap();
    let y = Array1::from(vec![5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.000_000_1]);

    let model = RANSACRegressor::new(LinearRegression::<f64>::new())
        .with_max_trials(200)
        .with_random_state(0);
    let fitted = model.fit(&x, &y).expect("RANSAC fit");

    // sklearn: MAD = 0 => threshold 0 => idx 7 (residual 1e-7) is an OUTLIER.
    const SK_MASK: [bool; 8] = [true, true, true, true, true, true, true, false];
    let mask = fitted.inlier_mask();
    assert_eq!(
        mask, &SK_MASK,
        "MAD-zero: sklearn threshold = MAD(y) = 0.0 marks idx 7 (residual 1e-7) \
         an outlier {SK_MASK:?}; ferrolearn substitutes 1e-6 and marks it an \
         inlier, reporting {mask:?}"
    );
}
