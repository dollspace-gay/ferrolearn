//! Divergence audit: `ferrolearn-model-sel::transformed_target::TransformedTargetRegressor`
//! vs scikit-learn 1.5.2 `sklearn.compose.TransformedTargetRegressor`
//! (`sklearn/compose/_target.py:24-356`).
//!
//! ferrolearn does not have `ferrolearn-linear` as a dev-dependency, so a real
//! `LinearRegression` inner regressor is NOT reachable from this crate's tests.
//! Per the task contract, the inner regressor here is a mean-predictor
//! (`MeanEstimator`), which is the exact behavioral analog of sklearn's
//! `DummyRegressor(strategy="mean")`. Every expected value below is computed by a
//! LIVE sklearn 1.5.2 oracle call (run from /tmp) wrapping THAT same
//! `DummyRegressor(strategy="mean")`, so the parity remains oracle-grounded
//! (goal.md R-CHAR-3) — never copied from the ferrolearn side.
//!
//! Live oracle (sklearn 1.5.2):
//! ```text
//! import numpy as np
//! from sklearn.dummy import DummyRegressor
//! from sklearn.compose import TransformedTargetRegressor
//! X = np.array([[1.],[2.],[3.],[4.]]); y = np.array([2.,5.,9.,16.])
//! tt = TransformedTargetRegressor(regressor=DummyRegressor(strategy="mean"),
//!                                 func=np.log, inverse_func=np.exp).fit(X, y)
//! tt.predict(X)  # -> 6.160140576482046 for every row (geometric mean)
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, Pipeline, PipelineEstimator};
use ferrolearn_core::traits::{Fit, Predict};
use ferrolearn_model_sel::TransformedTargetRegressor;
use ndarray::{Array1, Array2, array};

// ---------------------------------------------------------------------------
// Mean-predictor inner estimator == sklearn DummyRegressor(strategy="mean").
// Fits to mean(y); predicts that constant for every row. This is the exact
// behavioral analog used to oracle-ground the func/inverse value parity.
// ---------------------------------------------------------------------------

struct MeanEstimator;

impl PipelineEstimator<f64> for MeanEstimator {
    fn fit_pipeline(
        &self,
        _x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<Box<dyn FittedPipelineEstimator<f64>>, FerroError> {
        let mean = y.sum() / y.len() as f64;
        Ok(Box::new(FittedMeanEstimator { mean }))
    }
}

struct FittedMeanEstimator {
    mean: f64,
}

impl FittedPipelineEstimator<f64> for FittedMeanEstimator {
    fn predict_pipeline(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
        Ok(Array1::from_elem(x.nrows(), self.mean))
    }
}

fn mean_pipeline() -> Pipeline<f64> {
    Pipeline::new().estimator_step("mean", Box::new(MeanEstimator))
}

// ---------------------------------------------------------------------------
// REQ-1 GREEN GUARD: core func/inverse value parity vs the live oracle.
//
// Mirrors sklearn func-mode `regressor.fit(X, func(y))` /
// `inverse_func(regressor.predict(X))` (`sklearn/compose/_target.py:34-48`,
// `:274-288`, `:316-320`). Inner regressor == DummyRegressor(strategy="mean").
//
// Oracle (sklearn 1.5.2, DummyRegressor mean inner, func=np.log, inverse=np.exp,
// X=[[1],[2],[3],[4]], y=[2,5,9,16]): predict(X) == 6.160140576482046 for every
// row (the geometric mean exp(mean(log(y)))). NOT copied from ferrolearn.
//
// Should PASS now (inner mean-predictor matches sklearn's mean DummyRegressor).
// ---------------------------------------------------------------------------
#[test]
fn green_guard_log_exp_value_parity_vs_oracle() {
    // Expected, from the live sklearn 1.5.2 oracle (see module header).
    const SKLEARN_TTR_DUMMYMEAN_LOGEXP: f64 = 6.160140576482046;

    let ttr = TransformedTargetRegressor::new(mean_pipeline(), |y: f64| y.ln(), |y: f64| y.exp());
    let x = array![[1.0], [2.0], [3.0], [4.0]];
    let y = array![2.0, 5.0, 9.0, 16.0];

    let fitted = ttr.fit(&x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();

    assert_eq!(preds.len(), 4);
    for &p in preds.iter() {
        assert!(
            (p - SKLEARN_TTR_DUMMYMEAN_LOGEXP).abs() < 1e-9,
            "ferrolearn predicted {p}, sklearn oracle expects {SKLEARN_TTR_DUMMYMEAN_LOGEXP}"
        );
    }
}

// ---------------------------------------------------------------------------
// REQ-8 GREEN GUARD: sanctioned R-DEV-4 NaN-on-func deviation.
//
// ferrolearn `fit` returns `FerroError::NumericalInstability` when `func`
// produces NaN (`transformed_target.rs` fit; design REQ-8). sklearn also rejects
// a NaN func OUTPUT, but through `FunctionTransformer(validate=True)` re-running
// `check_array` (raising ValueError "Input y contains NaN."), NOT a finiteness
// guard on the original input. The OBSERVABLE outcome (an error) is the same;
// ferrolearn's error TYPE is a documented Rust footgun-elimination choice.
// Should PASS now.
// ---------------------------------------------------------------------------
#[test]
fn green_guard_nan_func_returns_err() {
    // func maps finite input y to NaN: sqrt(y-10) is NaN for y<10.
    let ttr = TransformedTargetRegressor::new(
        mean_pipeline(),
        |y: f64| (y - 10.0).sqrt(),
        |y: f64| y * y + 10.0,
    );
    let x = array![[1.0], [2.0], [3.0], [4.0]];
    let y = array![2.0, 5.0, 9.0, 16.0];

    let res = ttr.fit(&x, &y);
    let is_numerical_instability = matches!(res, Err(FerroError::NumericalInstability { .. }));
    assert!(
        is_numerical_instability,
        "expected FerroError::NumericalInstability for NaN func output"
    );
}

// ---------------------------------------------------------------------------
// DIVERGENCE (FAILING): infinite func output is NOT guarded by ferrolearn.
//
// sklearn (`sklearn/compose/_target.py:274` -> FunctionTransformer.transform ->
// `check_array(force_all_finite=True)`) rejects an INFINITE transformed target
// with `ValueError: "Input y contains infinity or a value too large for
// dtype('float64')."` (live oracle below). ferrolearn's `fit`
// (`transformed_target.rs`) guards ONLY `v.is_nan()`, not `v.is_infinite()`, so
// `1/y` on a y containing 0 yields `+inf`, which flows into the inner pipeline
// and produces an inf prediction instead of an error.
//
// Live oracle (sklearn 1.5.2):
//   X=[[1],[2],[3],[4]], y=[2,5,0,16], func=lambda z:1/z, inverse=lambda z:1/z,
//   check_inverse=False
//   -> raises ValueError "Input y contains infinity or a value too large ..."
//
// sklearn: errors. ferrolearn: returns Ok (no finiteness guard).
// Tracking: #1683 (specific blocker), parent #1682.
// ---------------------------------------------------------------------------
#[test]
#[ignore = "divergence: ferrolearn fit does not reject infinite func output (only NaN); sklearn check_array errors; tracking #1683"]
fn divergence_inf_func_output_not_rejected() {
    let ttr = TransformedTargetRegressor::new(mean_pipeline(), |y: f64| 1.0 / y, |y: f64| 1.0 / y);
    let x = array![[1.0], [2.0], [3.0], [4.0]];
    let y = array![2.0, 5.0, 0.0, 16.0]; // 1/0 -> +inf in transformed target

    let res = ttr.fit(&x, &y);
    // sklearn raises ValueError ("Input y contains infinity ...") at this point.
    // ferrolearn must reject the infinite transformed target too.
    assert!(
        res.is_err(),
        "sklearn rejects infinite transformed-target values; ferrolearn returned Ok"
    );
}
