//! Adversarial divergence audit of `validation_curve`'s per-cell error_score
//! handling against scikit-learn 1.5.2, focused on a NEW divergence introduced
//! by the #1758 fix (commit 26bf0fa8).
//!
//! The #1758 fix wrapped the per-`(param, fold)` fit/predict/score block in ONE
//! fallible closure and, on ANY `Err`, sets BOTH the train and test scores to
//! `f64::NAN`. That correctly mirrors sklearn for a FIT failure, but it COUPLES
//! the train and test scores: a failure in TRAIN scoring (or train predict) now
//! also clobbers the TEST score, and vice versa.
//!
//! scikit-learn scores train and test INDEPENDENTLY. In `_fit_and_score`
//! (`sklearn/model_selection/_validation.py:910-917`) the test score is computed
//! by one `_score(...)` call (`:910`) and the train score by a SEPARATE
//! `_score(...)` call (`:915`). `_score` (`:959-981`) catches a single-scorer
//! exception and substitutes `error_score` (np.nan) for THAT set ONLY:
//!
//! ```text
//! test_scores  = _score(estimator, X_test,  y_test,  scorer, ..., error_score)  # :910
//! train_scores = _score(estimator, X_train, y_train, scorer, ..., error_score)  # :915
//! ```
//!
//! So when scoring fails on the TRAIN fold but succeeds on the TEST fold, sklearn
//! returns `train == nan` and `test == FINITE`. ferrolearn nan-fills BOTH.
//!
//! All expected values are oracle-derived from the LIVE sklearn 1.5.2
//! `validation_curve` (never copied from the ferrolearn side, R-CHAR-3).

use ferrolearn_core::FerroError;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, Pipeline, PipelineEstimator};
use ferrolearn_model_sel::{KFold, validation_curve};
use ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// Constant-predicting estimator (ferrolearn analog of sklearn's
/// `parameters={param_name: v}` set on a clone). fit/predict NEVER fail.
struct ConstantEstimator {
    value: f64,
}
struct FittedConstant {
    value: f64,
}

impl PipelineEstimator<f64> for ConstantEstimator {
    fn fit_pipeline(
        &self,
        _x: &Array2<f64>,
        _y: &Array1<f64>,
    ) -> Result<Box<dyn FittedPipelineEstimator<f64>>, FerroError> {
        Ok(Box::new(FittedConstant { value: self.value }))
    }
}
impl FittedPipelineEstimator<f64> for FittedConstant {
    fn predict_pipeline(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
        Ok(Array1::from_elem(x.nrows(), self.value))
    }
}

/// Scoring that RAISES only on the TRAIN fold and succeeds on the TEST fold.
///
/// For `KFold(3)` over 30 samples, every train fold has 20 rows and every test
/// fold has 10 rows (verified identical to sklearn `KFold(3)`). We discriminate
/// the train fold by `y_true.len() == 20`, exactly mirroring the live-oracle
/// `make_scorer` that raises when `len(y_true) == 20`. On the test fold this
/// returns neg-MSE, the same metric ferrolearn computes elsewhere.
fn neg_mse_raise_on_train(
    y_true: &Array1<f64>,
    y_pred: &Array1<f64>,
) -> Result<f64, FerroError> {
    if y_true.len() == 20 {
        return Err(FerroError::InvalidParameter {
            name: "train_scorer".into(),
            reason: "scorer boom on train fold".into(),
        });
    }
    let diff = y_true - y_pred;
    Ok(-diff.mapv(|v| v * v).mean().unwrap_or(0.0))
}

// ---------------------------------------------------------------------------
// PINNED DIVERGENCE — train/test score independence under error_score=np.nan
// ---------------------------------------------------------------------------

/// Divergence: ferrolearn's `validation_curve`
/// (`ferrolearn-model-sel/src/validation_curve.rs:156-168`) scores train and
/// test inside ONE closure
///   `let cell = (|| { ...; let train = scoring(train..)?; ...; let test = scoring(test..)?; Ok((train,test)) })();`
///   `let (train, test) = cell.unwrap_or((f64::NAN, f64::NAN));`
/// so a TRAIN-scoring error (`scoring(&y_train, &y_train_pred)?` at line 160)
/// short-circuits the closure and forces BOTH train AND test to NaN.
///
/// scikit-learn scores them INDEPENDENTLY in `_fit_and_score`
/// (`sklearn/model_selection/_validation.py:910` test `_score`, `:915` train
/// `_score`); `_score` (`:967-981`) substitutes `error_score` for the FAILING
/// set ONLY. So a train-scoring failure leaves the TEST score FINITE.
///
/// LIVE sklearn 1.5.2 oracle (mirrors this exact fixture):
/// ```text
/// def my_score(y_true, y_pred):
///     if len(y_true) == 20: raise ValueError('train boom')   # train fold
///     return -np.mean((y_true - y_pred)**2)
/// scorer = make_scorer(my_score, greater_is_better=True)
/// class Const(BaseEstimator, RegressorMixin):
///     def __init__(self, d=1.0): self.d = d
///     def fit(self, X, y): return self
///     def predict(self, X): return np.full(X.shape[0], self.d)
/// tr, te = validation_curve(Const(), np.zeros((30,2)), np.arange(30.0),
///                           param_name='d', param_range=[1.0],
///                           cv=KFold(3), scoring=scorer, error_score=np.nan)
/// # tr.ravel() == [nan, nan, nan]            -> train all NaN
/// # te.ravel() == [-20.5, -190.5, -560.5]    -> test ALL FINITE
/// ```
///
/// sklearn: train all-NaN, test all-FINITE. ferrolearn: train AND test all-NaN
/// (the test score sklearn would have kept is clobbered). This test asserts the
/// sklearn behavior and therefore FAILS against the #1758 implementation.
///
/// Tracking: #1762 (blocker). This requires the GENERATOR to fix: split the
/// single closure into two independent train/test catch-and-continue scopes so a
/// train-side failure does not clobber the (sklearn-finite) test score.
// #1762
#[test]
#[ignore = "divergence: train-scoring failure clobbers test score (coupled closure); sklearn scores independently; tracking #1762"]
fn divergence_train_score_failure_clobbers_test_score() {
    let y: Array1<f64> = (0..30).map(f64::from).collect();
    let x = Array2::<f64>::zeros((30, 2));
    let kf = KFold::new(3);

    let result = validation_curve(
        &x,
        &y,
        &kf,
        &[1.0],
        |val| Pipeline::new().estimator_step("est", Box::new(ConstantEstimator { value: val })),
        neg_mse_raise_on_train,
    )
    .expect("sklearn returns a partial nan-bearing curve (train nan, test finite), not an error");

    assert_eq!(result.train_scores.shape(), &[1, 3], "shape (n_params, n_folds)");
    assert_eq!(result.test_scores.shape(), &[1, 3], "shape (n_params, n_folds)");

    // sklearn nan-fills ONLY the failing (train) set.
    for j in 0..3usize {
        assert!(
            result.train_scores[[0, j]].is_nan(),
            "train fold {j}: scorer raised, sklearn sets error_score=np.nan, got {}",
            result.train_scores[[0, j]],
        );
    }

    // KEY divergence assertion: sklearn keeps the TEST score FINITE because train
    // and test are scored by SEPARATE _score calls. ferrolearn's coupled closure
    // clobbers it to NaN.
    for j in 0..3usize {
        assert!(
            result.test_scores[[0, j]].is_finite(),
            "test fold {j}: sklearn keeps test FINITE (train-scoring failure must \
             not clobber the test score); got {}",
            result.test_scores[[0, j]],
        );
    }
}
