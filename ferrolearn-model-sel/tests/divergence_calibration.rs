//! Adversarial divergence audit of `ferrolearn-model-sel::calibration` against
//! scikit-learn 1.5.2 `sklearn.calibration` / `sklearn.isotonic`.
//!
//! All expected values are computed from the LIVE sklearn 1.5.2 oracle
//! (R-CHAR-3 — never copied from the ferrolearn side); the oracle snippet that
//! produced each constant is quoted in the doc-comment above the test.
//!
//! ## How the calibrators are isolated through the PUBLIC API
//!
//! `fit_sigmoid` / `fit_isotonic` / `isotonic_lookup` are private. They are
//! reached deterministically through `CalibratedClassifierCV::fit` + `predict`
//! using an **identity-score base estimator**: a `FitFn` that ignores its
//! training inputs and returns a `PredictFn` mapping `X -> X.column(0)`.
//!
//! Consequences that make the OOF flow deterministic:
//!   * Each sample `i` appears in exactly one validation fold, and its
//!     out-of-fold score is exactly `X[i, 0]`. So the aggregated OOF
//!     `(score, label)` multiset equals `{(X[i,0], y[i])}` — a permutation of
//!     the inputs.
//!   * Both calibrators are order-independent: sigmoid sums over samples,
//!     isotonic sorts by score. So the fitted calibrator equals
//!     `fit_sigmoid(scores, labels)` / `fit_isotonic(scores, labels)` on the
//!     ORIGINAL inputs regardless of fold partition.
//!   * `predict` re-runs the identity base on the full query matrix, so
//!     `predict(Q)[i] == calibrator.transform(Q[i,0])` — i.e. the calibrator
//!     evaluated at the query score.

use ferrolearn_core::{FerroError, Predict};
use ferrolearn_model_sel::calibration::{
    CalibratedClassifierCV, CalibrationMethod, FitFn, PredictFn,
};
use ndarray::{Array1, Array2};

/// A `FitFn` whose returned predictor reproduces `X.column(0)` as the raw
/// decision score for every sample (identity-score base estimator).
fn identity_score_fit_fn() -> FitFn {
    Box::new(|_x: &Array2<f64>, _y: &Array1<usize>| {
        Ok(
            Box::new(|x: &Array2<f64>| -> Result<Array1<f64>, FerroError> {
                Ok(x.column(0).to_owned())
            }) as PredictFn,
        )
    })
}

/// Build an `(n, 1)` feature matrix whose column 0 holds `scores`.
fn scores_as_x(scores: &[f64]) -> Array2<f64> {
    Array2::from_shape_vec((scores.len(), 1), scores.to_vec()).unwrap()
}

/// Fit a `CalibratedClassifierCV` with the identity-score base on
/// `(scores, labels)` and evaluate the resulting calibrator at `queries`.
fn calibrate_and_predict(
    method: CalibrationMethod,
    cv: usize,
    scores: &[f64],
    labels: &[usize],
    queries: &[f64],
) -> Vec<f64> {
    let x = scores_as_x(scores);
    let y = Array1::from_vec(labels.to_vec());
    let cal = CalibratedClassifierCV::new(identity_score_fit_fn(), method, cv);
    let fitted = cal.fit(&x, &y).unwrap();
    let q = scores_as_x(queries);
    fitted.predict(&q).unwrap().to_vec()
}

// ===========================================================================
// REQ-SIGMOID — GREEN GUARD (would FAIL if Platt scaling regressed)
// ===========================================================================

/// Green guard for REQ-SIGMOID (SHIPPED). ferrolearn's `fit_sigmoid` (Newton)
/// must reproduce sklearn `_sigmoid_calibration` (L-BFGS-B) calibrated
/// probabilities `expit(-(A*f+B))` on `|F|<30` data — same convex objective,
/// same Platt targets, sign reparam `a=-A, b=-B`.
///
/// LIVE ORACLE (sklearn 1.5.2, run from /tmp):
/// ```text
/// from sklearn.calibration import _sigmoid_calibration
/// from scipy.special import expit
/// scores=np.array([-2.,-1.,0.5,1.,2.,3.,-1.5,0.2,2.5,-0.7])
/// labels=np.array([0,0,0,1,1,1,0,1,1,0])
/// A,B=_sigmoid_calibration(scores,labels)   # A=-0.9352810706, B=0.3392523752
/// expit(-(A*scores+B)) ->
///   [0.098872597944,0.218482188704,0.532053023243,0.644747209666,
///    0.822197867808,0.921766272382,0.149037009005,0.4620242363,
///    0.880686822297,0.270132757661]
/// ```
#[test]
fn green_sigmoid_well_separated_matches_oracle() {
    let scores = [-2.0, -1.0, 0.5, 1.0, 2.0, 3.0, -1.5, 0.2, 2.5, -0.7];
    let labels = [0, 0, 0, 1, 1, 1, 0, 1, 1, 0];
    // Query at the same points (predict re-applies the calibrator).
    let got = calibrate_and_predict(CalibrationMethod::Sigmoid, 5, &scores, &labels, &scores);

    // Oracle: sklearn expit(-(A*f+B)) at the fitted (A,B).
    let oracle = [
        0.098_872_597_944,
        0.218_482_188_704,
        0.532_053_023_243,
        0.644_747_209_666,
        0.822_197_867_808,
        0.921_766_272_382,
        0.149_037_009_005,
        0.462_024_236_3,
        0.880_686_822_297,
        0.270_132_757_661,
    ];
    for (i, (&g, &o)) in got.iter().zip(oracle.iter()).enumerate() {
        assert!(
            (g - o).abs() < 1e-4,
            "sigmoid prob[{i}] ferro={g} sklearn={o} diff={}",
            (g - o).abs()
        );
    }
}

/// Green guard for REQ-SIGMOID on a HARDER mixed (non-separable) case — labels
/// interleave with scores so the convex minimum is non-trivial. Still `|F|<30`.
///
/// LIVE ORACLE (sklearn 1.5.2, run from /tmp):
/// ```text
/// s2=np.array([-2.5,-1.,0.,0.3,1.2,2.,-0.5,0.8,1.5,-1.8,2.2,-0.2])
/// l2=np.array([0,1,0,1,0,1,0,0,1,0,1,1])
/// A2,B2=_sigmoid_calibration(s2,l2)   # A2=-0.5259237607, B2=0.0965898410
/// expit(-(A2*s2+B2)) ->
///   [0.196013452857,0.349209988044,0.47587129609,0.515292051126,
///    0.630536402366,0.722171309776,0.411068085891,0.580335101924,
///    0.666477444154,0.260522888317,0.742775085651,0.449726800431]
/// ```
#[test]
fn green_sigmoid_mixed_nonseparable_matches_oracle() {
    let scores = [
        -2.5, -1.0, 0.0, 0.3, 1.2, 2.0, -0.5, 0.8, 1.5, -1.8, 2.2, -0.2,
    ];
    let labels = [0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1];
    let got = calibrate_and_predict(CalibrationMethod::Sigmoid, 4, &scores, &labels, &scores);

    let oracle = [
        0.196_013_452_857,
        0.349_209_988_044,
        0.475_871_296_09,
        0.515_292_051_126,
        0.630_536_402_366,
        0.722_171_309_776,
        0.411_068_085_891,
        0.580_335_101_924,
        0.666_477_444_154,
        0.260_522_888_317,
        0.742_775_085_651,
        0.449_726_800_431,
    ];
    for (i, (&g, &o)) in got.iter().zip(oracle.iter()).enumerate() {
        assert!(
            (g - o).abs() < 1e-4,
            "sigmoid prob[{i}] ferro={g} sklearn={o} diff={}",
            (g - o).abs()
        );
    }
}

// ===========================================================================
// REQ-ISOTONIC — DIVERGENCE PIN (#1800)
// ===========================================================================

/// Divergence: ferrolearn isotonic calibrator diverges from
/// `sklearn/calibration.py:670` (`IsotonicRegression(out_of_bounds="clip")`).
///
/// ferrolearn `fit_isotonic` (calibration.rs:352) emits one breakpoint per PAV
/// block at the block MIDPOINT `f64::midpoint(lo, hi)` (calibration.rs:401) and
/// `isotonic_lookup` (calibration.rs:410) linearly interpolates between those
/// midpoints. sklearn interpolates over the ACTUAL unique sorted X-thresholds
/// with pooled PAV means.
///
/// LIVE ORACLE (sklearn 1.5.2, run from /tmp):
/// ```text
/// from sklearn.isotonic import IsotonicRegression
/// s=np.array([0.,1.,2.,3.,4.,5.,2.5,1.5,3.5]); y=np.array([0,0,1,0,1,1,0,1,1],dtype=float)
/// ir=IsotonicRegression(out_of_bounds='clip').fit(s,y)
/// ir.X_thresholds_ -> [0.0, 1.0, 1.5, 3.0, 3.5, 5.0]
/// ir.y_thresholds_ -> [0.0, 0.0, 0.5, 0.5, 1.0, 1.0]
/// q=np.array([-1.,0.5,1.2,2.2,2.7,3.3,4.4,6.])
/// ir.predict(q) -> [0., 0., 0.2, 0.5, 0.5, 0.8, 1., 1.]
/// ```
/// ferrolearn produces `[0, 0, 0.08, 0.48, 0.68, 0.92, 1, 1]` (midpoint
/// breakpoints `[0,1,2.25,3.5,4,5]`) — MAX ABS DIFF 0.18.
/// Tracking: #1800
#[test]
fn divergence_isotonic_breakpoints_1800() {
    let scores = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 2.5, 1.5, 3.5];
    let labels = [0, 0, 1, 0, 1, 1, 0, 1, 1];
    let queries = [-1.0, 0.5, 1.2, 2.2, 2.7, 3.3, 4.4, 6.0];
    let got = calibrate_and_predict(CalibrationMethod::Isotonic, 3, &scores, &labels, &queries);

    // sklearn IsotonicRegression(out_of_bounds="clip").predict(q)
    let oracle = [0.0, 0.0, 0.2, 0.5, 0.5, 0.8, 1.0, 1.0];
    for (i, (&g, &o)) in got.iter().zip(oracle.iter()).enumerate() {
        assert!(
            (g - o).abs() < 1e-6,
            "isotonic prob[{i}] (q={}) ferro={g} sklearn={o} diff={}",
            queries[i],
            (g - o).abs()
        );
    }
}

// ===========================================================================
// REQ-ISOTONIC — NEW DIVERGENCE PIN: tied scores not pre-averaged
// ===========================================================================

/// Divergence: ferrolearn isotonic calibrator diverges from
/// `sklearn/isotonic.py:319` (`unique_X, unique_y, unique_sample_weight =
/// _make_unique(X, y, sample_weight)`) reached via
/// `sklearn/calibration.py:670` (`IsotonicRegression(out_of_bounds="clip")`).
///
/// sklearn collapses tied-X samples into ONE point carrying the (weighted)
/// AVERAGE y BEFORE running PAV (`isotonic.py:322`,
/// `y = isotonic_regression(unique_y, ...)`). ferrolearn `fit_isotonic`
/// (calibration.rs:381) pushes each raw sample as its own PAV block and never
/// pre-averages duplicate scores, so tied scores with mixed labels produce a
/// different mapping.
///
/// Input: scores=[1,1,2,2] labels=[0,1,0,1] (two ties, each split 0/1).
/// sklearn `_make_unique` -> X=[1,2] y=[0.5,0.5]; already monotone ->
/// X_thresholds_=[1,2] y_thresholds_=[0.5,0.5] -> predict == 0.5 everywhere.
/// ferrolearn keeps four PAV inputs [0,1,0,1] at scores [1,1,2,2], merges the
/// trailing violator into one block but still ramps from the leading 0-block,
/// yielding a non-flat mapping.
///
/// LIVE ORACLE (sklearn 1.5.2, run from /tmp):
/// ```text
/// from sklearn.isotonic import IsotonicRegression
/// import numpy as np
/// s=np.array([1.,1.,2.,2.]); y=np.array([0,1,0,1],dtype=float)
/// ir=IsotonicRegression(out_of_bounds='clip').fit(s,y)
/// ir.X_thresholds_ -> [1.0, 2.0]
/// ir.y_thresholds_ -> [0.5, 0.5]
/// q=np.array([1.,1.25,1.5,1.75,2.]); ir.predict(q) -> [0.5,0.5,0.5,0.5,0.5]
/// ```
/// ferrolearn produces `[0.0, 0.5, 0.5, 0.5, 1.0]` at the same queries
/// (MAX ABS DIFF 0.5 at q=1.0 and q=2.0).
/// Tracking: // #NNNN new-blocker
#[ignore = "divergence: isotonic does not pre-average tied scores (_make_unique); tracking #NNNN"]
#[test]
fn divergence_isotonic_tied_scores_make_unique() {
    let scores = [1.0, 1.0, 2.0, 2.0];
    let labels = [0, 1, 0, 1];
    let queries = [1.0, 1.25, 1.5, 1.75, 2.0];
    let got = calibrate_and_predict(CalibrationMethod::Isotonic, 2, &scores, &labels, &queries);

    // sklearn IsotonicRegression(out_of_bounds="clip").predict(q) — flat 0.5
    // because _make_unique averages each tied X to y=0.5 before PAV.
    let oracle = [0.5, 0.5, 0.5, 0.5, 0.5];
    for (i, (&g, &o)) in got.iter().zip(oracle.iter()).enumerate() {
        assert!(
            (g - o).abs() < 1e-6,
            "isotonic prob[{i}] (q={}) ferro={g} sklearn={o} diff={}",
            queries[i],
            (g - o).abs()
        );
    }
}
