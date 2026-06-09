//! Adversarial divergence audit (#2365) of `cross_val_score` /
//! `cross_validate` / `cross_val_predict` per-fold mechanics using a REAL
//! DETERMINISTIC estimator (Ridge), pushing beyond the MeanEstimator coverage
//! in `divergence_cross_validation.rs`.
//!
//! Every expected value below is a LIVE sklearn 1.5.2 oracle value (recorded
//! inline next to the assertion), NEVER copied from the ferrolearn side
//! (R-CHAR-3). Fixture (deterministic, replicated exactly in Rust):
//!
//! ```python
//! X = np.array([[float(i), float((i*7)%5)] for i in range(12)])
//! y = np.array([2.0*i - 3.0*((i*7)%5) + 1.0 for i in range(12)])
//! est = Ridge(alpha=1.0); kf = KFold(4)
//! cross_val_score(est, X, y, cv=kf, scoring='r2')
//!   -> [0.9879436469258941, 0.99777697232161, 0.99777697232161, 0.9929727858241079]
//! cross_val_score(est, X, y, cv=kf, scoring='neg_mean_squared_error')
//!   -> [-0.12860109945712953, -0.045942572020060336, -0.04594257202005886, -0.14522909296843814]
//! cross_val_score(est, X, y, cv=kf, scoring='neg_mean_absolute_error')
//!   -> [-0.2987494040415151, -0.2003167062549497, -0.17462831002023252, -0.31269870955676043]
//! cross_val_score(est, X, y)  # default cv=5, scoring=r2(estimator.score)
//!   -> [0.9879436469258941, 0.99777697232161, 0.9844755514144126, 0.9959359494555504, 0.949846012101589]
//! cross_val_predict(est, X, y, cv=kf)
//!   -> [0.9073605457145977, -2.763010232148753, -6.433381010012104, 3.884402216943785,
//!       0.1848772763262092, 10.699524940617575, 7.038532594352072, 3.339007653734491,
//!       13.853655318025865, 9.949971946886105, 20.41658874135029, 16.695343183093325]
//! ```

use ferrolearn_core::pipeline::Pipeline;
use ferrolearn_linear::Ridge;
use ferrolearn_model_sel::{KFold, cross_val_predict, cross_val_score, cross_validate};
use ndarray::{Array1, Array2};

const TOL: f64 = 1e-7;

fn fixture() -> (Array2<f64>, Array1<f64>) {
    let mut x = Array2::<f64>::zeros((12, 2));
    let mut y = Array1::<f64>::zeros(12);
    for i in 0..12 {
        let f0 = i as f64;
        let f1 = ((i * 7) % 5) as f64;
        x[[i, 0]] = f0;
        x[[i, 1]] = f1;
        y[i] = 2.0 * f0 - 3.0 * f1 + 1.0;
    }
    (x, y)
}

fn ridge_pipeline() -> Pipeline {
    Pipeline::new().estimator_step("ridge", Box::new(Ridge::<f64>::new().with_alpha(1.0)))
}

/// sklearn `r2_score` (`sklearn/metrics/_regression.py:1146`):
/// `1 - SS_res/SS_tot`, `SS_tot = sum((y - mean(y))^2)`.
fn r2(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64, ferrolearn_core::FerroError> {
    let mean = y_true.mean().unwrap_or(0.0);
    let ss_res: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| (t - p).powi(2))
        .sum();
    let ss_tot: f64 = y_true.iter().map(|t| (t - mean).powi(2)).sum();
    Ok(1.0 - ss_res / ss_tot)
}

/// scorer 'neg_mean_squared_error' (`sklearn/metrics/_scorer.py`): sign = -1.
fn neg_mse(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64, ferrolearn_core::FerroError> {
    let diff = y_true - y_pred;
    Ok(-(diff.mapv(|v| v * v).mean().unwrap_or(0.0)))
}

/// scorer 'neg_mean_absolute_error' (`sklearn/metrics/_scorer.py`): sign = -1.
fn neg_mae(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64, ferrolearn_core::FerroError> {
    let diff = y_true - y_pred;
    Ok(-(diff.mapv(f64::abs).mean().unwrap_or(0.0)))
}

/// CORE per-fold scores with a real deterministic estimator (Ridge, alpha=1.0),
/// KFold(4), scoring='r2'. Live sklearn 1.5.2 oracle (see header). Verifies the
/// SCORES + the fold ORDER. Mirrors `cross_validate -> test_score`
/// (`sklearn/model_selection/_validation.py:560`).
#[test]
fn cvs_ridge_kfold4_r2() {
    let (x, y) = fixture();
    let scores = cross_val_score(&ridge_pipeline(), &x, &y, &KFold::new(4), r2).unwrap();
    let expected = [
        0.9879436469258941_f64,
        0.99777697232161,
        0.99777697232161,
        0.9929727858241079,
    ];
    assert_eq!(scores.len(), 4);
    for (i, (got, exp)) in scores.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < TOL,
            "fold {i} r2 diverged: ferrolearn {got}, sklearn {exp}"
        );
    }
}

/// neg_mean_squared_error sign + values. The scores MUST be NEGATIVE
/// (scorer applies sign=-1 to MSE). Live sklearn oracle (see header).
#[test]
fn cvs_ridge_kfold4_negmse_sign() {
    let (x, y) = fixture();
    let scores = cross_val_score(&ridge_pipeline(), &x, &y, &KFold::new(4), neg_mse).unwrap();
    let expected = [
        -0.12860109945712953_f64,
        -0.045942572020060336,
        -0.04594257202005886,
        -0.14522909296843814,
    ];
    for (i, (got, exp)) in scores.iter().zip(expected.iter()).enumerate() {
        assert!(*got < 0.0, "neg_mse fold {i} must be negative, got {got}");
        assert!(
            (got - exp).abs() < TOL,
            "fold {i} neg_mse diverged: ferrolearn {got}, sklearn {exp}"
        );
    }
}

/// neg_mean_absolute_error sign + values. Live sklearn oracle (see header).
#[test]
fn cvs_ridge_kfold4_negmae_sign() {
    let (x, y) = fixture();
    let scores = cross_val_score(&ridge_pipeline(), &x, &y, &KFold::new(4), neg_mae).unwrap();
    let expected = [
        -0.2987494040415151_f64,
        -0.2003167062549497,
        -0.17462831002023252,
        -0.31269870955676043,
    ];
    for (i, (got, exp)) in scores.iter().zip(expected.iter()).enumerate() {
        assert!(*got < 0.0, "neg_mae fold {i} must be negative, got {got}");
        assert!(
            (got - exp).abs() < TOL,
            "fold {i} neg_mae diverged: ferrolearn {got}, sklearn {exp}"
        );
    }
}

/// DEFAULT cv. sklearn `cross_val_score(est, X, y)` with no `cv` uses cv=5
/// (KFold for a regressor, via `check_cv`, `_validation.py:560` -> `_split.py`).
/// ferrolearn's `cross_val_score` has NO default — `cv` is mandatory — so the
/// closest equivalent is `KFold::new(5)`. This pins that the FIVE-fold r2 scores
/// (which a default-cv `cross_val_score` would have to produce) match the live
/// sklearn DEFAULT oracle `cross_val_score(est, X, y)`.
#[test]
fn cvs_ridge_default_cv5_r2() {
    let (x, y) = fixture();
    let scores = cross_val_score(&ridge_pipeline(), &x, &y, &KFold::new(5), r2).unwrap();
    // Live sklearn DEFAULT (no cv arg) oracle:
    let expected = [
        0.9879436469258941_f64,
        0.99777697232161,
        0.9844755514144126,
        0.9959359494555504,
        0.949846012101589,
    ];
    assert_eq!(scores.len(), 5, "sklearn default cv=5 => 5 folds");
    for (i, (got, exp)) in scores.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < TOL,
            "default-cv fold {i} r2 diverged: ferrolearn {got}, sklearn {exp}"
        );
    }
}

/// cross_validate test_score must match cross_val_score (same per-fold loop).
/// Live sklearn oracle (KFold(4), r2 — see header).
#[test]
fn cvalidate_ridge_kfold4_r2_matches() {
    let (x, y) = fixture();
    let r = cross_validate(&ridge_pipeline(), &x, &y, &KFold::new(4), r2, false).unwrap();
    let expected = [
        0.9879436469258941_f64,
        0.99777697232161,
        0.99777697232161,
        0.9929727858241079,
    ];
    for (i, (got, exp)) in r.test_scores.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < TOL,
            "cross_validate fold {i} r2 diverged: ferrolearn {got}, sklearn {exp}"
        );
    }
}

/// cross_val_predict out-of-fold predictions with Ridge, KFold(4). Each sample
/// is predicted by the model trained on the OTHER folds, placed at its original
/// index. Live sklearn oracle (see header). Mirrors `_validation.py:1054`.
#[test]
fn cvp_ridge_kfold4() {
    let (x, y) = fixture();
    let preds = cross_val_predict(&ridge_pipeline(), &x, &y, &KFold::new(4)).unwrap();
    let expected = [
        0.9073605457145977_f64,
        -2.763010232148753,
        -6.433381010012104,
        3.884402216943785,
        0.1848772763262092,
        10.699524940617575,
        7.038532594352072,
        3.339007653734491,
        13.853655318025865,
        9.949971946886105,
        20.41658874135029,
        16.695343183093325,
    ];
    assert_eq!(preds.len(), 12);
    for (i, (got, exp)) in preds.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < TOL,
            "OOF prediction at index {i} diverged: ferrolearn {got}, sklearn {exp}"
        );
    }
}

/// Clone-per-fold (no leakage), edge cv=2. Each fold fits a FRESH Ridge on its
/// own train split; the two folds train on DISJOINT halves and yield DISTINCT
/// r2 values that each independently match sklearn. If fold 2 reused fold 1's
/// fitted coefficients (leakage), the scores would not match the live oracle.
/// Live sklearn oracle: `cross_val_score(Ridge(1.0), X, y, cv=KFold(2),
/// scoring='r2') -> [0.9945850786134888, 0.9869113773547142]`.
#[test]
fn cvs_ridge_cv2_clone_independence() {
    let (x, y) = fixture();
    let scores = cross_val_score(&ridge_pipeline(), &x, &y, &KFold::new(2), r2).unwrap();
    let expected = [0.9945850786134888_f64, 0.9869113773547142];
    assert_eq!(scores.len(), 2);
    for (i, (got, exp)) in scores.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < TOL,
            "cv2 fold {i} r2 diverged (possible leakage): ferrolearn {got}, sklearn {exp}"
        );
    }
    // The two folds train on disjoint data => distinct scores (leakage guard).
    assert!(
        (scores[0] - scores[1]).abs() > 1e-6,
        "the two folds must be independent (distinct scores)"
    );
}
