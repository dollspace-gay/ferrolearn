//! Divergence + green-guard tests for `GridSearchCV` / `CvResults` against
//! scikit-learn 1.5.2 `sklearn.model_selection.GridSearchCV` / `BaseSearchCV`
//! (`sklearn/model_selection/_search.py`).
//!
//! - GREEN guards pin the SHIPPED behaviors (REQ-1 exhaustive mechanic + grid
//!   order, REQ-2 unweighted fold mean). They PASS today and would FAIL if the
//!   claim regressed.
//! - `#[ignore]`'d FAILING pins capture deterministic divergences:
//!     * `best_index_first_on_tie` — REQ-BESTIDX, tracking #1776.
//!     * `best_index_nan_mean_tied_worst` — UNCLAIMED NaN-mean divergence,
//!       tracking #1782.
//!
//! All expected values come from the LIVE sklearn 1.5.2 oracle (run from /tmp),
//! never copied from the ferrolearn side (R-CHAR-3).

use ferrolearn_core::FerroError;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, Pipeline, PipelineEstimator};
use ferrolearn_model_sel::{GridSearchCV, KFold, ParamSet, ParamValue, param_grid};
use ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// Estimator that always predicts a fixed constant value.
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

/// Estimator that predicts the training mean (mirrors sklearn's `Mean`
/// dummy used in the AC-2 oracle).
struct MeanEstimator;
struct FittedMean {
    mean: f64,
}
impl PipelineEstimator<f64> for MeanEstimator {
    fn fit_pipeline(
        &self,
        _x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<Box<dyn FittedPipelineEstimator<f64>>, FerroError> {
        Ok(Box::new(FittedMean {
            mean: y.mean().unwrap_or(0.0),
        }))
    }
}
impl FittedPipelineEstimator<f64> for FittedMean {
    fn predict_pipeline(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
        Ok(Array1::from_elem(x.nrows(), self.mean))
    }
}

/// Negative MSE (higher is better), matching sklearn `neg_mean_squared_error`.
fn neg_mse(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64, FerroError> {
    let diff = y_true - y_pred;
    Ok(-diff.mapv(|v| v * v).mean().unwrap_or(0.0))
}

fn const_factory(key: &'static str) -> impl Fn(&ParamSet) -> Pipeline {
    move |params: &ParamSet| {
        let val = match params.get(key) {
            Some(ParamValue::Float(v)) => *v,
            _ => 0.0,
        };
        Pipeline::new().estimator_step("est", Box::new(ConstantEstimator { value: val }))
    }
}

// ---------------------------------------------------------------------------
// GREEN guard — REQ-1: exhaustive mechanic + grid order
// ---------------------------------------------------------------------------

/// Green guard: REQ-1 — every candidate is cross-validated and recorded, in
/// ParameterGrid (sorted-key) order, with `n_folds` per-fold scores each.
///
/// Oracle (live sklearn 1.5.2, from /tmp):
/// ```text
/// class Const(BaseEstimator, RegressorMixin): ... predict -> full(n, c)
/// X=zeros((30,2)); y=full(30,1.0)
/// gs=GridSearchCV(Const(), {'c':[0.0,1.0,2.0]},
///     scoring='neg_mean_squared_error', cv=KFold(3), refit=False).fit(X,y)
/// len(gs.cv_results_['params'])           -> 3
/// [p['c'] for p in gs.cv_results_['params']] -> [0.0, 1.0, 2.0]
/// ```
/// Mirrors `_run_search` -> `evaluate_candidates(ParameterGrid(...))`
/// (`sklearn/model_selection/_search.py:1571-1573`, `:952-1017`).
#[test]
fn green_req1_exhaustive_mechanic_and_order() {
    let grid = param_grid! { "c" => [0.0_f64, 1.0_f64, 2.0_f64] };
    let mut gs = GridSearchCV::new(
        Box::new(const_factory("c")),
        grid,
        Box::new(KFold::new(3)),
        neg_mse,
    );
    let x = Array2::<f64>::zeros((30, 2));
    let y = Array1::<f64>::from_elem(30, 1.0);
    gs.fit(&x, &y).unwrap();

    let results = gs.cv_results().unwrap();

    // Oracle: len(gs.cv_results_['params']) == 3.
    const SK_N_CANDIDATES: usize = 3;
    assert_eq!(results.params.len(), SK_N_CANDIDATES);
    assert_eq!(results.mean_scores.len(), SK_N_CANDIDATES);
    assert_eq!(results.all_scores.len(), SK_N_CANDIDATES);

    // Oracle: each candidate scored on n_splits=3 folds.
    const SK_N_FOLDS: usize = 3;
    for fold_scores in &results.all_scores {
        assert_eq!(fold_scores.len(), SK_N_FOLDS);
    }

    // Oracle: candidate order == [0.0, 1.0, 2.0] (ParameterGrid sorted-key).
    const SK_C_ORDER: [f64; 3] = [0.0, 1.0, 2.0];
    let ferro_order: Vec<f64> = results
        .params
        .iter()
        .map(|p| match p.get("c") {
            Some(ParamValue::Float(v)) => *v,
            other => panic!("expected Float c, got {other:?}"),
        })
        .collect();
    assert_eq!(ferro_order.as_slice(), &SK_C_ORDER);
}

// ---------------------------------------------------------------------------
// GREEN guard — REQ-2: unweighted fold mean
// ---------------------------------------------------------------------------

/// Green guard: REQ-2 — `mean_test_score` is the UNWEIGHTED arithmetic mean of
/// the per-fold scores (`np.average(array, axis=1, weights=None)`,
/// `sklearn/model_selection/_search.py:1097`).
///
/// Oracle (live sklearn 1.5.2, from /tmp):
/// ```text
/// class Mean(BaseEstimator, RegressorMixin): fit -> m_=y.mean(); predict -> full(n,m_)
/// X=zeros((9,1)); y=arange(9.0)
/// gs=GridSearchCV(Mean(), {'off':[0.0]}, scoring='neg_mean_squared_error',
///     cv=KFold(3), refit=False).fit(X,y)
/// splits = [-20.916666666666668, -0.6666666666666666, -20.916666666666668]
/// gs.cv_results_['mean_test_score'][0] -> -14.166666666666666
/// ```
/// The per-fold scores genuinely vary, so this is not a constant-fold no-op:
/// the mean must equal the simple average. Cross-checked against the oracle
/// constant AND against ferrolearn's own per-fold scores (the actual contract).
#[test]
fn green_req2_unweighted_fold_mean() {
    let grid = param_grid! { "off" => [0.0_f64] };
    let mut gs = GridSearchCV::new(
        Box::new(|_p: &ParamSet| Pipeline::new().estimator_step("mean", Box::new(MeanEstimator))),
        grid,
        Box::new(KFold::new(3)),
        neg_mse,
    );
    let x = Array2::<f64>::zeros((9, 1));
    let y: Array1<f64> = (0..9).map(|i| i as f64).collect();
    gs.fit(&x, &y).unwrap();

    let results = gs.cv_results().unwrap();
    let mean = results.mean_scores[0];
    let folds = &results.all_scores[0];

    // Contract: mean == simple average of this candidate's own per-fold scores.
    let simple_avg = folds.sum() / folds.len() as f64;
    assert!(
        (mean - simple_avg).abs() < 1e-12,
        "mean {mean} != simple average {simple_avg} of folds {folds:?}"
    );

    // Cross-check vs the live sklearn oracle constant.
    const SK_MEAN_TEST_SCORE: f64 = -14.166_666_666_666_666;
    assert!(
        (mean - SK_MEAN_TEST_SCORE).abs() < 1e-12,
        "mean {mean} != sklearn mean_test_score {SK_MEAN_TEST_SCORE}"
    );
}

// ---------------------------------------------------------------------------
// FAILING pin — REQ-BESTIDX: first-on-tie (#1776)
// ---------------------------------------------------------------------------

/// Divergence: `CvResults::best_index` (`grid_search.rs:71-77`) uses
/// `mean_scores.iter().enumerate().max_by(partial_cmp)`. Rust `Iterator::max_by`
/// returns the LAST element among equals, so on a TIE for the best mean score it
/// selects the HIGHEST-index candidate.
///
/// sklearn `best_index_ = results["rank_test_score"].argmin()`
/// (`sklearn/model_selection/_search.py:840`) over
/// `rankdata(-array_means, method="min")` (`:1129`) — `np.argmin` returns the
/// FIRST occurrence of the minimum, so the LOWEST-index tied candidate wins.
///
/// Oracle (live sklearn 1.5.2, from /tmp): grid `{'c':[4.0,6.0]}`, constant
/// `y == 5.0`, each MSE == 1 -> tied means [-1.0, -1.0]:
/// ```text
/// gs.best_index_              -> 0
/// gs.best_params_['c']        -> 4.0
/// gs.cv_results_['rank_test_score'].tolist() -> [1, 1]
/// ```
/// ferrolearn returns `Some(1)` (c == 6.0). Tracking #1776.
///
/// NOTE: the divergent `best_index` is SHARED — `RandomizedSearchCV` and the
/// halving searches also consume it, so the fix benefits all of them; pinned
/// here only via grid_search.
#[test]
fn best_index_first_on_tie() {
    let grid = param_grid! { "c" => [4.0_f64, 6.0_f64] };
    let mut gs = GridSearchCV::new(
        Box::new(const_factory("c")),
        grid,
        Box::new(KFold::new(3)),
        neg_mse,
    );
    let x = Array2::<f64>::zeros((30, 2));
    let y = Array1::<f64>::from_elem(30, 5.0);
    gs.fit(&x, &y).unwrap();

    let results = gs.cv_results().unwrap();
    // Sanity: the means really are tied (both -1.0); otherwise this is not a
    // tie test.
    assert!(
        (results.mean_scores[0] - results.mean_scores[1]).abs() < 1e-12,
        "precondition: means must tie, got {:?}",
        results.mean_scores
    );

    // Oracle: sklearn best_index_ == 0, best_params_['c'] == 4.0.
    const SK_BEST_INDEX: usize = 0;
    const SK_BEST_C: f64 = 4.0;
    assert_eq!(
        results.best_index(),
        Some(SK_BEST_INDEX),
        "sklearn picks first-on-tie (index 0); ferrolearn max_by picks last"
    );
    assert_eq!(
        gs.best_params().unwrap().get("c"),
        Some(&ParamValue::Float(SK_BEST_C))
    );
}

// ---------------------------------------------------------------------------
// FAILING pin — UNCLAIMED NaN-mean divergence (new blocker #1782)
// ---------------------------------------------------------------------------

/// Divergence (UNCLAIMED): when a candidate's mean score is NaN (a NaN-bearing
/// fold makes `scores.mean()` NaN), sklearn treats it as TIED-WORST and never
/// lets it win:
/// `min_array_means = np.nanmin(array_means) - 1;
///  array_means = np.nan_to_num(array_means, nan=min_array_means);
///  rank = rankdata(-array_means, method="min")`
/// (`sklearn/model_selection/_search.py:1127-1129`). So a finite candidate
/// always outranks a NaN candidate, regardless of position.
///
/// ferrolearn `CvResults::best_index` (`grid_search.rs:71-77`) compares means
/// with `a.partial_cmp(b).unwrap_or(Ordering::Equal)` inside `max_by`. A NaN
/// comparison yields `Equal`, so a NaN can be retained as the running max when
/// it appears AFTER the finite candidate -> the NaN candidate is reported as
/// best.
///
/// This NaN mean IS reachable: `cross_val_score` does not reject NaN
/// predictions/scores, and `CvResults::push` keeps a NaN mean as-is (only an
/// EMPTY fold set maps to NEG_INFINITY).
///
/// Oracle (live sklearn 1.5.2, from /tmp): grid order finite-then-NaN
/// `{'c':[5.0, 999.0]}` (999 -> NaN predictions), constant `y == 5.0`:
/// ```text
/// gs.cv_results_['mean_test_score'] -> [0.0, nan]
/// gs.cv_results_['rank_test_score'].argmin() -> 0   (the finite candidate)
/// ```
/// ferrolearn `best_index()` on means `[0.0, NaN]` returns `Some(1)` (the NaN
/// candidate). Tracking #1782 (new-blocker).
#[test]
fn best_index_nan_mean_tied_worst() {
    // Estimator that predicts NaN when the param flags it, otherwise a constant.
    struct MaybeNan {
        c: f64,
    }
    struct FittedMaybeNan {
        c: f64,
    }
    impl PipelineEstimator<f64> for MaybeNan {
        fn fit_pipeline(
            &self,
            _x: &Array2<f64>,
            _y: &Array1<f64>,
        ) -> Result<Box<dyn FittedPipelineEstimator<f64>>, FerroError> {
            Ok(Box::new(FittedMaybeNan { c: self.c }))
        }
    }
    impl FittedPipelineEstimator<f64> for FittedMaybeNan {
        fn predict_pipeline(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
            let v = if self.c == 999.0 { f64::NAN } else { self.c };
            Ok(Array1::from_elem(x.nrows(), v))
        }
    }

    // Grid order: finite candidate FIRST (c=5.0), NaN candidate LAST (c=999.0).
    // This is the position where ferrolearn's max_by lets the NaN survive.
    let grid = param_grid! { "c" => [5.0_f64, 999.0_f64] };
    let mut gs = GridSearchCV::new(
        Box::new(|params: &ParamSet| {
            let val = match params.get("c") {
                Some(ParamValue::Float(v)) => *v,
                _ => 0.0,
            };
            Pipeline::new().estimator_step("est", Box::new(MaybeNan { c: val }))
        }),
        grid,
        Box::new(KFold::new(3)),
        neg_mse,
    );
    let x = Array2::<f64>::zeros((30, 2));
    let y = Array1::<f64>::from_elem(30, 5.0);
    gs.fit(&x, &y).unwrap();

    let results = gs.cv_results().unwrap();
    // Precondition: candidate 0 finite (0.0), candidate 1 NaN.
    assert!(
        results.mean_scores[0].is_finite() && results.mean_scores[1].is_nan(),
        "precondition: means must be [finite, NaN], got {:?}",
        results.mean_scores
    );

    // Oracle: sklearn ranks the finite candidate best -> best_index 0.
    const SK_BEST_INDEX: usize = 0;
    assert_eq!(
        results.best_index(),
        Some(SK_BEST_INDEX),
        "sklearn treats NaN mean as tied-worst (finite candidate wins); \
         ferrolearn max_by lets a trailing NaN win"
    );
}
