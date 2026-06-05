//! Adversarial divergence audit of `validation_curve` against scikit-learn 1.5.2
//! `sklearn.model_selection.validation_curve`
//! (`sklearn/model_selection/_validation.py:2149-2316`).
//!
//! GREEN guards pin SHIPPED behavior (REQ-1 orientation, REQ-2 train scores,
//! REQ-3 shape, REQ-4 vary-param) so a future regression would FAIL them.
//! `#[ignore]`'d tests pin genuine, single-file-fixable behavioral divergences
//! against the LIVE sklearn oracle (REQ-7 error_score=np.nan continue, #1758).
//!
//! All expected values are oracle-derived (live sklearn 1.5.2 or a pure-numpy
//! reproduction of sklearn's `.reshape(-1, n_params).T` algebra), never copied
//! from the ferrolearn side (R-CHAR-3).

use ferrolearn_core::FerroError;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, Pipeline, PipelineEstimator};
use ferrolearn_model_sel::{KFold, validation_curve};
use ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// Estimator that ignores its training data and always predicts the constant
/// `value` (the swept parameter). This is the ferrolearn analog of sklearn's
/// `parameters={param_name: v}` set on a clone.
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

/// Estimator that fails to fit when `fail` is true — the ferrolearn analog of
/// sklearn's estimator raising in `fit` for one `param_range` value.
struct MaybeFailingEstimator {
    value: f64,
    fail: bool,
}

impl PipelineEstimator<f64> for MaybeFailingEstimator {
    fn fit_pipeline(
        &self,
        _x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<Box<dyn FittedPipelineEstimator<f64>>, FerroError> {
        if self.fail {
            return Err(FerroError::InvalidParameter {
                name: "boom".into(),
                reason: "estimator failed to fit for this param value".into(),
            });
        }
        Ok(Box::new(FittedMean {
            mean: y.mean().unwrap_or(0.0),
            value: self.value,
        }))
    }
}

struct FittedMean {
    mean: f64,
    #[allow(
        dead_code,
        reason = "carried for parity with sklearn fitted-param state"
    )]
    value: f64,
}

impl FittedPipelineEstimator<f64> for FittedMean {
    fn predict_pipeline(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
        Ok(Array1::from_elem(x.nrows(), self.mean))
    }
}

fn neg_mse(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64, FerroError> {
    let diff = y_true - y_pred;
    Ok(-diff.mapv(|v| v * v).mean().unwrap_or(0.0))
}

/// Scoring that encodes BOTH the fold (via `mean(y_true)`) AND the parameter
/// (via the constant prediction) into a single distinct, recoverable value:
/// `score = mean(y_true) * 10.0 + y_pred[0]`. Combined with a `y` whose value
/// equals its fold index and a `ConstantEstimator(param)`, cell `(param i, fold
/// j)` deterministically equals `j*10 + param_i`. This is the swap/transpose
/// detector for REQ-1.
fn fold_and_param_probe(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64, FerroError> {
    let fold_part = y_true.mean().unwrap_or(0.0) * 10.0;
    let param_part = y_pred[0];
    Ok(fold_part + param_part)
}

// ---------------------------------------------------------------------------
// GREEN GUARD — REQ-1: matrix orientation, end-to-end transpose/swap detector
// ---------------------------------------------------------------------------

/// REQ-1 (KEY). Verifies the `(param OUTER, fold INNER)` row-major fill produces
/// a `(n_params, n_folds)` matrix element-for-element IDENTICAL to sklearn's
/// `(fold OUTER, param INNER)` flat list `.reshape(-1, n_params).T`
/// (`sklearn/model_selection/_validation.py:2306-2314`:
/// `for train, test in cv.split(...) for v in param_range` then
/// `results["test_scores"].reshape(-1, n_params).T`).
///
/// Adversarial construction: ferrolearn's KFold(shuffle=False) splits 30 samples
/// into contiguous folds 0..10, 10..20, 20..30 (verified identical to sklearn's
/// `KFold(3)`). `y[row] = floor(row/10)` makes every test-fold's `y` constant
/// and equal to its fold index `j`. A `ConstantEstimator(param)` predicts
/// `param` on test. With `fold_and_param_probe`, cell `(param i, fold j)`
/// == `j*10 + param_i` — a DISTINCT value per cell that the same-value-per-cell
/// SHIPPED tests cannot distinguish from a transpose.
///
/// Oracle (pure-numpy reproduction of sklearn's reshape algebra, AC-1):
/// ```text
/// flat = [f*10+p for f in range(3) for p in range(4)]
/// M = flat.reshape(-1, 4).T   # M[p][f] == f*10 + p
/// ```
/// param_values = [0,1,2,3] so param_i == i.
#[test]
fn req1_orientation_end_to_end_transpose_detector() {
    // y[row] = floor(row/10) => fold j's contiguous test set has y == j.
    let y: Array1<f64> = (0..30).map(|r| f64::from(r / 10)).collect();
    let x = Array2::<f64>::zeros((30, 2));
    let kf = KFold::new(3);

    let result = validation_curve(
        &x,
        &y,
        &kf,
        &[0.0, 1.0, 2.0, 3.0],
        |val| Pipeline::new().estimator_step("est", Box::new(ConstantEstimator { value: val })),
        fold_and_param_probe,
    )
    .unwrap();

    assert_eq!(
        result.test_scores.shape(),
        &[4, 3],
        "shape must be (n_params, n_folds)"
    );

    // Oracle: cell (param i, fold j) == j*10 + i. Values derived from sklearn's
    // `.reshape(-1, n_params).T` algebra (numpy reproduction), NOT from ferrolearn.
    for i in 0..4usize {
        for j in 0..3usize {
            let expected = (j as f64) * 10.0 + (i as f64);
            let got = result.test_scores[[i, j]];
            assert!(
                (got - expected).abs() < 1e-9,
                "cell (param {i}, fold {j}): expected {expected} (= fold*10 + param), got {got}; \
                 a transpose or fold/param swap would surface here",
            );
        }
    }
}

// ---------------------------------------------------------------------------
// GREEN GUARD — REQ-2 / REQ-3: train scores always returned, shape & orientation
// ---------------------------------------------------------------------------

/// REQ-2 (`return_train_score=True` hardcoded,
/// `sklearn/model_selection/_validation.py:2303`) and REQ-3 (returned shape
/// `(n_ticks, n_cv_folds)`, `:2257-2261`/`:2313-2314`).
///
/// Live oracle (sklearn 1.5.2): `validation_curve(DecisionTreeRegressor(), X, y,
/// param_range=[1,2,3], cv=KFold(3))` returns `tr.shape == te.shape == (3, 3)`.
#[test]
fn req2_req3_train_scores_returned_with_shape() {
    let y: Array1<f64> = (0..30).map(f64::from).collect();
    let x = Array2::<f64>::zeros((30, 2));
    let kf = KFold::new(3);

    let result = validation_curve(
        &x,
        &y,
        &kf,
        &[1.0, 2.0, 3.0],
        |val| Pipeline::new().estimator_step("est", Box::new(ConstantEstimator { value: val })),
        neg_mse,
    )
    .unwrap();

    // sklearn returns train_scores unconditionally; ferrolearn must too.
    assert_eq!(
        result.train_scores.shape(),
        &[3, 3],
        "train_scores (n_ticks, n_folds)"
    );
    assert_eq!(
        result.test_scores.shape(),
        &[3, 3],
        "test_scores (n_ticks, n_folds)"
    );
    assert_eq!(
        result.train_scores.len(),
        9,
        "every (param,fold) train cell populated"
    );
    for &s in &result.train_scores {
        assert!(s.is_finite(), "train score must be finite, got {s}");
    }
}

// ---------------------------------------------------------------------------
// GREEN GUARD — REQ-4: vary-one-param mechanic threads the param into the model
// ---------------------------------------------------------------------------

/// REQ-4 (vary one param over a range via the `make_pipeline` closure — the
/// R-DEV-7 analog of sklearn's `clone(estimator)` + `set_params`,
/// `sklearn/model_selection/_validation.py:2292`/`:2299`).
///
/// With a constant `y == 1.0`, a `ConstantEstimator(param)` scored by neg_mse
/// has its best (== 0) test score exactly at `param == 1.0` and strictly worse
/// scores elsewhere — proving the swept param actually reaches the model and is
/// not ignored.
#[test]
fn req4_param_threads_into_model() {
    let y = Array1::<f64>::from_elem(30, 1.0);
    let x = Array2::<f64>::zeros((30, 2));
    let kf = KFold::new(3);

    let result = validation_curve(
        &x,
        &y,
        &kf,
        &[0.0, 1.0, 5.0],
        |val| Pipeline::new().estimator_step("est", Box::new(ConstantEstimator { value: val })),
        neg_mse,
    )
    .unwrap();

    let mean_at = |row: usize| {
        result.test_scores.row(row).iter().sum::<f64>() / result.test_scores.ncols() as f64
    };
    assert!(
        mean_at(1).abs() < 1e-10,
        "param=1.0 (== y) must be best (~0), got {}",
        mean_at(1)
    );
    assert!(
        mean_at(0) < -0.5,
        "param=0.0 must be strictly worse, got {}",
        mean_at(0)
    );
    assert!(
        mean_at(2) < -0.5,
        "param=5.0 must be strictly worse, got {}",
        mean_at(2)
    );
}

// ---------------------------------------------------------------------------
// PINNED DIVERGENCE — REQ-7: error_score=np.nan continue-the-curve (#1758)
// ---------------------------------------------------------------------------

/// Divergence: ferrolearn's `validation_curve`
/// (`ferrolearn-model-sel/src/validation_curve.rs:147` `pipeline.fit(...)?`)
/// diverges from `sklearn/model_selection/_validation.py:2304` (`_fit_and_score`
/// `error_score=error_score`, default `np.nan`, `:2243-2246`).
///
/// When the estimator fails to fit for ONE swept param value, sklearn fills that
/// `(fold, param)` row with `error_score` (np.nan), raises `FitFailedWarning`,
/// and CONTINUES the curve. ferrolearn propagates the failure via `?` and ABORTS
/// the whole call with `Err`, returning no partial curve.
///
/// LIVE sklearn 1.5.2 oracle (AC-7):
/// ```text
/// class F(BaseEstimator, RegressorMixin):
///     def __init__(self,d=1): self.d=d
///     def fit(self,X,y):
///         if self.d==2: raise ValueError('boom')
///         self.m_=y.mean(); return self
///     def predict(self,X): return np.full(X.shape[0], self.m_)
/// tr,te = validation_curve(F(), X, y, param_name='d', param_range=[1,2,3],
///                          cv=KFold(3), scoring='neg_mean_squared_error')
/// # te.shape == (3, 3); te[1] == [nan, nan, nan]; np.isnan(te).any() == True
/// ```
///
/// Here the middle param value (index 1) maps to a failing estimator. sklearn
/// returns a full `(3, 3)` matrix whose row 1 is all-NaN; ferrolearn returns
/// `Err`. This test asserts the sklearn behavior and therefore FAILS against
/// the current implementation.
///
/// Tracking: #1758. Pin assesses this as a single-file fix to
/// `validation_curve.rs` (a default error_score=NaN catch-and-continue around
/// the per-cell fit/score, replacing the `?` propagation) — see report for the
/// signature-vs-default-behavior judgment.
#[test]
fn req7_error_score_nan_continue() {
    let y: Array1<f64> = (0..30).map(f64::from).collect();
    let x = Array2::<f64>::zeros((30, 2));
    let kf = KFold::new(3);

    // param_values = [1.0, 2.0, 3.0]; the value 2.0 (index 1) fails to fit,
    // mirroring sklearn's `d==2` raising in `fit`.
    let result = validation_curve(
        &x,
        &y,
        &kf,
        &[1.0, 2.0, 3.0],
        |val| {
            let fail = (val - 2.0).abs() < 1e-12;
            Pipeline::new()
                .estimator_step("est", Box::new(MaybeFailingEstimator { value: val, fail }))
        },
        neg_mse,
    )
    .expect("sklearn returns a partial nan-bearing curve, not an error");

    assert_eq!(
        result.test_scores.shape(),
        &[3, 3],
        "shape (n_params, n_folds)"
    );

    // Oracle: the failing param row (index 1) is all-NaN; other rows are finite.
    for j in 0..3usize {
        assert!(
            result.test_scores[[1, j]].is_nan(),
            "failing-param row 1, fold {j}: sklearn fills with np.nan, got {}",
            result.test_scores[[1, j]],
        );
        assert!(
            result.test_scores[[0, j]].is_finite(),
            "non-failing param row 0 must remain finite",
        );
        assert!(
            result.test_scores[[2, j]].is_finite(),
            "non-failing param row 2 must remain finite",
        );
    }
}
