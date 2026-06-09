//! Exhaustive hyperparameter search with cross-validation.
//!
//! [`GridSearchCV`] evaluates every combination in a parameter grid using
//! cross-validation and records the results in a [`CvResults`] struct.
//!
//! # Example
//!
//! ```rust,no_run
//! use ferrolearn_model_sel::{GridSearchCV, KFold, param_grid, ParamValue};
//! use ferrolearn_core::pipeline::Pipeline;
//! use ferrolearn_core::FerroError;
//! use ndarray::{Array1, Array2};
//!
//! fn neg_mse(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64, FerroError> {
//!     let diff = y_true - y_pred;
//!     Ok(-diff.mapv(|v| v * v).mean().unwrap_or(0.0))
//! }
//!
//! // Pipeline factory using params (in practice would actually use them).
//! // let factory = |_params: &_| Pipeline::new().estimator_step("est", Box::new(MyEst));
//! // let grid = param_grid! { "alpha" => [0.1_f64, 1.0_f64] };
//! // let mut gs = GridSearchCV::new(Box::new(factory), grid, Box::new(KFold::new(3)), neg_mse);
//! // gs.fit(&x, &y).unwrap();
//! // println!("Best: {:?}", gs.best_params());
//! ```
//!
//! ## REQ status
//!
//! Mirrors `sklearn.model_selection.GridSearchCV` / `BaseSearchCV`
//! (`sklearn/model_selection/_search.py:1210` / `:436`, v1.5.2). Every REQ is
//! BINARY (R-DEFER-2): SHIPPED (end-to-end functional + non-test consumer + tests
//! + verification) or NOT-STARTED (with a concrete blocker).
//!
//! | REQ | Status | Notes |
//! |---|---|---|
//! | REQ-1 (exhaustive search mechanic) | SHIPPED | `fit` runs `cross_val_score` for every `ParamSet` in grid order (= `ParameterGrid` sorted-key order); mirrors `evaluate_candidates(ParameterGrid(...))` (`:1571`). Factory closure = R-DEV-7 analog of `clone(estimator).set_params`. Guard `green_req1_exhaustive_mechanic_and_order`. |
//! | REQ-2 (mean_test_score = unweighted fold mean) | SHIPPED | `CvResults::push` mean = `scores.mean()` = `np.average(array, axis=1, weights=None)` (`:1097`). Guard `green_req2_unweighted_fold_mean`. |
//! | REQ-BESTIDX (best_index = rank argmin: first-on-tie, NaN tied-worst) | SHIPPED | `CvResults::best_index` returns the max mean with strict-`>` (first index wins ties) and NaN treated as worse than any finite (all-NaN ⇒ index 0) — reproducing `best_index_ = rank_test_score.argmin()` over `rankdata(-means, "min")` with `nan_to_num` (`:840`, `:1123-1129`). Was `max_by` (LAST on tie, trailing-NaN could win). Tests `best_index_first_on_tie` (#1776), `best_index_nan_mean_tied_worst` (#1782). |
//! | REQ-REFIT (refit + best_estimator_ + delegating predict/score) | NOT-STARTED | search-only struct; sklearn `refit=True` default refits best params on full data and delegates predict/score (`:1046-1061`, `:577`). Blocker #1777. |
//! | REQ-CVRESULTS (std_test_score + rank_test_score) | SHIPPED | `CvResults` now carries `std_scores` (population std ddof=0 over folds, `fn population_std` consumed by `CvResults::push`) = `cv_results_["std_test_score"]` (`:1112-1117`) and `rank_scores` (`fn rankdata_min_neg` consumed by `CvResults::finalize_ranks`, called from `GridSearchCV::fit`) = `rankdata(-mean, method="min")` with NaN→tied-worst / all-NaN→ones (`:1119-1132`). Getters `std_test_score()`/`rank_test_score()`. Consumer: `GridSearchCV::fit` populates both; `CvResults` re-exported in `lib.rs`. Tests `divergence_grid_search_cvresults_2367.rs` (live Ridge oracle ~1e-7 std + exact ranks incl. a min-tie). |
//! | REQ-CVRESULTS-REST (split{i}_test_score keyed / fit-score times / param_<name>) | NOT-STARTED | `CvResults` still lacks per-split keyed access, fit/score timing, and `param_<name>` masked-array columns (`:1095`,`:1134-1139`). Blocker #1778. |
//! | REQ-DEFAULT-CV (default cv=None ⇒ 5-fold classifier-aware) | NOT-STARTED | `cv` mandatory; no `check_cv` / StratifiedKFold dispatch (`:928`). Blocker #1779. |
//! | REQ-DEFAULT-SCORING (default scoring=None ⇒ estimator scorer) | NOT-STARTED | `scoring` mandatory; no `check_scoring` r2/accuracy default (`:857`). Blocker #1779. |
//! | REQ-PARALLEL (n_jobs/pre_dispatch/verbose/return_train_score/multimetric) | NOT-STARTED | none exposed (`:1210` init). Blocker #1780. |
//! | REQ-ERROR-SCORE (error_score=np.nan continue) | NOT-STARTED | CROSS-UNIT — `fit` delegates to `cross_val_score`, which `?`-propagates; sklearn nan-fills a failing (candidate, split) cell and continues (`:996`). Owned by `cross_validation.rs` per S8, not a grid_search blocker. |
//! | REQ-X-1 (R-SUBSTRATE) | NOT-STARTED | `ndarray::{Array1, Array2}`; destination `ferray-core` (R-SUBSTRATE-1). Blocker #1781. |
//! | REQ-X-2 (non-test production consumer) | SHIPPED | re-exported `pub use grid_search::{CvResults, GridSearchCV}` in `lib.rs`; `CvResults` further consumed by `random_search.rs`, `halving_grid_search.rs`, `halving_random_search.rs` (the `best_index` fix benefits all). |

use ferrolearn_core::FerroError;
use ferrolearn_core::pipeline::Pipeline;
use ndarray::{Array1, Array2};

use crate::cross_validation::{CrossValidator, cross_val_score};
use crate::param_grid::ParamSet;

// ---------------------------------------------------------------------------
// CvResults
// ---------------------------------------------------------------------------

/// Results collected during a [`GridSearchCV`] or
/// [`crate::RandomizedSearchCV`] run.
///
/// Each entry corresponds to one parameter combination that was evaluated.
#[derive(Debug, Clone)]
pub struct CvResults {
    /// The parameter sets that were evaluated, in evaluation order.
    pub params: Vec<ParamSet>,
    /// Mean cross-validation score for each parameter set.
    pub mean_scores: Vec<f64>,
    /// Population standard deviation (ddof=0) of the per-fold scores for each
    /// parameter set — mirrors `cv_results_["std_test_score"]`.
    pub std_scores: Vec<f64>,
    /// All per-fold scores for each parameter set.
    pub all_scores: Vec<Array1<f64>>,
    /// Competition rank of each parameter set by mean score (1 = best),
    /// with `method="min"` tie handling — mirrors
    /// `cv_results_["rank_test_score"]`. Populated by [`GridSearchCV::fit`]
    /// (and the other search drivers) once every candidate has been recorded.
    pub rank_scores: Vec<usize>,
}

impl CvResults {
    pub(crate) fn new() -> Self {
        Self {
            params: Vec::new(),
            mean_scores: Vec::new(),
            std_scores: Vec::new(),
            all_scores: Vec::new(),
            rank_scores: Vec::new(),
        }
    }

    pub(crate) fn push(&mut self, params: ParamSet, scores: Array1<f64>) {
        let mean = scores.mean().unwrap_or(f64::NEG_INFINITY);
        let std = population_std(&scores, mean);
        self.params.push(params);
        self.mean_scores.push(mean);
        self.std_scores.push(std);
        self.all_scores.push(scores);
    }

    /// Recompute [`rank_scores`](CvResults::rank_scores) from the recorded
    /// [`mean_scores`](CvResults::mean_scores).
    ///
    /// Mirrors `cv_results_["rank_test_score"]`
    /// (`sklearn/model_selection/_search.py:1119-1132`):
    /// `rankdata(-array_means, method="min")` after NaN means are mapped to
    /// tied-worst (`nan_to_num(array_means, nan=nanmin-1)`); an all-NaN mean
    /// vector yields all-ones. Call once after every candidate has been
    /// [`push`](CvResults::push)ed.
    pub(crate) fn finalize_ranks(&mut self) {
        self.rank_scores = rankdata_min_neg(&self.mean_scores);
    }

    /// Return the index of the parameter set with the highest mean score.
    ///
    /// Mirrors scikit-learn's `best_index_ = results["rank_test_score"].argmin()`
    /// (`sklearn/model_selection/_search.py:840`) over
    /// `rankdata(-array_means, method="min")` with NaN means mapped to tied-worst
    /// (`nan_to_num(array_means, nan=nanmin-1)`, `:1127-1129`). Concretely:
    ///
    /// - Ties on the maximum mean are broken by the FIRST (lowest) index, since
    ///   `np.argmin` returns the first occurrence of rank 1 (strict improvement
    ///   only replaces the running best).
    /// - A NaN mean is treated as worse than any finite mean: a finite candidate
    ///   always outranks a NaN one regardless of position, and a NaN never beats
    ///   a finite one.
    /// - If every mean is NaN, all ranks are 1, so `argmin` is index 0.
    ///
    /// Returns `None` if no results have been recorded.
    pub fn best_index(&self) -> Option<usize> {
        if self.mean_scores.is_empty() {
            return None;
        }
        let mut best_i = 0usize;
        let mut best = self.mean_scores[0];
        for (i, &m) in self.mean_scores.iter().enumerate().skip(1) {
            let better = match (best.is_nan(), m.is_nan()) {
                (true, false) => true,      // finite candidate beats a NaN running-best
                (false, true) => false,     // NaN candidate never beats a finite best
                (true, true) => false,      // both NaN: keep the earlier index
                (false, false) => m > best, // strict > ⇒ first index wins on ties
            };
            if better {
                best = m;
                best_i = i;
            }
        }
        Some(best_i)
    }

    /// Population standard deviation (ddof=0) of the per-fold scores for each
    /// parameter set — `cv_results_["std_test_score"]`.
    ///
    /// Mirrors `array_stds = np.sqrt(np.average((array - array_means[:, None])**2,
    /// axis=1, weights=None))` (`sklearn/model_selection/_search.py:1112-1117`):
    /// the unweighted (population, ddof=0) std over the per-fold test scores.
    #[must_use]
    pub fn std_test_score(&self) -> &[f64] {
        &self.std_scores
    }

    /// Competition rank of each parameter set by mean score, 1 = best, with
    /// `method="min"` tie handling — `cv_results_["rank_test_score"]`.
    ///
    /// Mirrors `rankdata(-array_means, method="min").astype(np.int32)`
    /// (`sklearn/model_selection/_search.py:1129`): tied means share the same
    /// (lowest) rank and the next distinct value skips by the tie-group size.
    /// Empty until [`finalize_ranks`](CvResults::finalize_ranks) has run.
    #[must_use]
    pub fn rank_test_score(&self) -> &[usize] {
        &self.rank_scores
    }
}

/// Population standard deviation (ddof=0) of `scores` given their `mean`.
///
/// Mirrors `np.sqrt(np.average((array - array_means[:, None])**2, axis=1))`
/// (`sklearn/model_selection/_search.py:1112-1117`) — the NumPy/`np.std`
/// default divides by `N`, not `N-1`. An empty fold set yields `0.0`
/// (no panic; `np.average` of an empty array would raise, but the search
/// always produces at least one fold per candidate).
fn population_std(scores: &Array1<f64>, mean: f64) -> f64 {
    let n = scores.len();
    if n == 0 {
        return 0.0;
    }
    let sum_sq: f64 = scores.iter().map(|&s| (s - mean) * (s - mean)).sum();
    (sum_sq / n as f64).sqrt()
}

/// `rankdata(-values, method="min")` with sklearn's NaN handling.
///
/// Returns the competition ranks of `-values` (so rank 1 = the LARGEST value)
/// using scipy's `method="min"` semantics: every value receives a rank equal to
/// `1 + (the number of values strictly greater than it)`, so tied values share
/// the lowest rank of their group and the next distinct value's rank skips by
/// the tie-group size.
///
/// NaN handling mirrors `sklearn/model_selection/_search.py:1119-1132`: a NaN
/// mean is treated as tied-worst (`nan_to_num(values, nan=nanmin-1)`), so any
/// finite value outranks it; if EVERY value is NaN, all ranks are `1` (sklearn
/// short-circuits to `np.ones_like`).
fn rankdata_min_neg(values: &[f64]) -> Vec<usize> {
    let n = values.len();
    if n == 0 {
        return Vec::new();
    }
    // sklearn: all-NaN means → all-ones rank vector (no ranking performed).
    if values.iter().all(|v| v.is_nan()) {
        return vec![1usize; n];
    }
    // `nan_to_num(values, nan=nanmin-1)`: NaN means become strictly worse than
    // every finite mean, so they sink to the worst (highest) rank.
    let nanmin = values
        .iter()
        .copied()
        .filter(|v| !v.is_nan())
        .fold(f64::INFINITY, f64::min);
    let filled: Vec<f64> = values
        .iter()
        .map(|&v| if v.is_nan() { nanmin - 1.0 } else { v })
        .collect();
    // rankdata(-filled, method="min"): rank = 1 + count of values strictly
    // greater than this one (larger mean ⇒ smaller rank ⇒ better).
    filled
        .iter()
        .map(|&v| 1 + filled.iter().filter(|&&o| o > v).count())
        .collect()
}

// ---------------------------------------------------------------------------
// GridSearchCV
// ---------------------------------------------------------------------------

/// Exhaustive search over a parameter grid using cross-validation.
///
/// For every combination in `param_grid`, [`GridSearchCV`] builds a pipeline
/// via the user-supplied factory closure, runs cross-validation, and records
/// the results.  After calling [`fit`](GridSearchCV::fit) the best parameters
/// and score can be retrieved via the accessor methods.
///
/// # Type Parameters
///
/// The struct is generic over the factory lifetime `'a`.
///
/// # Example
///
/// ```rust,no_run
/// use ferrolearn_model_sel::{GridSearchCV, KFold, param_grid, ParamValue};
/// use ferrolearn_core::pipeline::{Pipeline, PipelineEstimator, FittedPipelineEstimator};
/// use ferrolearn_core::FerroError;
/// use ndarray::{Array1, Array2};
///
/// struct ConstantEstimator(f64);
/// struct FittedConstant(f64);
///
/// impl PipelineEstimator<f64> for ConstantEstimator {
///     fn fit_pipeline(&self, _x: &Array2<f64>, _y: &Array1<f64>)
///         -> Result<Box<dyn FittedPipelineEstimator<f64>>, FerroError>
///     {
///         Ok(Box::new(FittedConstant(self.0)))
///     }
/// }
/// impl FittedPipelineEstimator<f64> for FittedConstant {
///     fn predict_pipeline(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
///         Ok(Array1::from_elem(x.nrows(), self.0))
///     }
/// }
///
/// fn neg_mse(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64, FerroError> {
///     let diff = y_true - y_pred;
///     Ok(-diff.mapv(|v| v * v).mean().unwrap_or(0.0))
/// }
///
/// let factory = |params: &ferrolearn_model_sel::ParamSet| {
///     let val = match params.get("constant") {
///         Some(ParamValue::Float(v)) => *v,
///         _ => 0.0,
///     };
///     Pipeline::new().estimator_step("est", Box::new(ConstantEstimator(val)))
/// };
/// let grid = param_grid! { "constant" => [0.0_f64, 1.0_f64, 2.0_f64] };
/// let mut gs = GridSearchCV::new(Box::new(factory), grid, Box::new(KFold::new(3)), neg_mse);
///
/// let x = Array2::<f64>::zeros((30, 1));
/// let y = Array1::<f64>::from_elem(30, 1.0);
/// gs.fit(&x, &y).unwrap();
/// println!("{:?}", gs.best_params());
/// ```
pub struct GridSearchCV<'a> {
    /// Factory that builds a [`Pipeline`] from a [`ParamSet`].
    pipeline_factory: Box<dyn Fn(&ParamSet) -> Pipeline + 'a>,
    /// All parameter combinations to try.
    param_grid: Vec<ParamSet>,
    /// Cross-validator used to evaluate each combination.
    cv: Box<dyn CrossValidator>,
    /// Scoring function: `(y_true, y_pred) -> Result<f64>`. Higher is better.
    scoring: fn(&Array1<f64>, &Array1<f64>) -> Result<f64, FerroError>,
    /// Results populated after [`fit`](GridSearchCV::fit) is called.
    results: Option<CvResults>,
}

impl<'a> GridSearchCV<'a> {
    /// Create a new [`GridSearchCV`].
    ///
    /// # Parameters
    ///
    /// - `pipeline_factory` — closure that accepts a [`ParamSet`] and returns
    ///   an unfitted [`Pipeline`].
    /// - `param_grid` — the parameter combinations to search over.
    /// - `cv` — the cross-validator (e.g., [`KFold`](crate::KFold)).
    /// - `scoring` — scoring function; higher values are considered better.
    pub fn new(
        pipeline_factory: Box<dyn Fn(&ParamSet) -> Pipeline + 'a>,
        param_grid: Vec<ParamSet>,
        cv: Box<dyn CrossValidator>,
        scoring: fn(&Array1<f64>, &Array1<f64>) -> Result<f64, FerroError>,
    ) -> Self {
        Self {
            pipeline_factory,
            param_grid,
            cv,
            scoring,
            results: None,
        }
    }

    /// Run the grid search.
    ///
    /// Iterates over all parameter combinations, builds a pipeline for each,
    /// runs cross-validation, and stores the results internally.
    ///
    /// # Errors
    ///
    /// Returns a [`FerroError`] if the parameter grid is empty, if any pipeline
    /// fails to fit, or if the cross-validator fails.
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<(), FerroError> {
        if self.param_grid.is_empty() {
            return Err(FerroError::InvalidParameter {
                name: "param_grid".into(),
                reason: "parameter grid must not be empty".into(),
            });
        }

        let mut results = CvResults::new();

        for params in &self.param_grid {
            let pipeline = (self.pipeline_factory)(params);
            let scores = cross_val_score(&pipeline, x, y, self.cv.as_ref(), self.scoring)?;
            results.push(params.clone(), scores);
        }
        // rank_test_score depends on the full set of means, so it is computed
        // once every candidate has been recorded (mirrors `_format_results`
        // ranking after the per-candidate means are assembled,
        // `sklearn/model_selection/_search.py:1119-1132`).
        results.finalize_ranks();

        self.results = Some(results);
        Ok(())
    }

    /// Return a reference to the full cross-validation results.
    ///
    /// Returns `None` if [`fit`](GridSearchCV::fit) has not been called.
    pub fn cv_results(&self) -> Option<&CvResults> {
        self.results.as_ref()
    }

    /// Return the parameter set that achieved the highest mean score.
    ///
    /// Returns `None` if [`fit`](GridSearchCV::fit) has not been called.
    pub fn best_params(&self) -> Option<&ParamSet> {
        let results = self.results.as_ref()?;
        let idx = results.best_index()?;
        results.params.get(idx)
    }

    /// Return the best mean cross-validation score.
    ///
    /// Returns `None` if [`fit`](GridSearchCV::fit) has not been called.
    pub fn best_score(&self) -> Option<f64> {
        let results = self.results.as_ref()?;
        let idx = results.best_index()?;
        results.mean_scores.get(idx).copied()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ferrolearn_core::pipeline::{FittedPipelineEstimator, Pipeline, PipelineEstimator};
    use ndarray::{Array1, Array2};

    use crate::KFold;
    use crate::param_grid::ParamValue;

    // -----------------------------------------------------------------------
    // Test fixtures
    // -----------------------------------------------------------------------

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

    /// Estimator that predicts the training mean (for regression accuracy).
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

    /// Negative MSE scoring (higher is better).
    fn neg_mse(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64, FerroError> {
        let diff = y_true - y_pred;
        Ok(-diff.mapv(|v| v * v).mean().unwrap_or(0.0))
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_grid_search_runs_all_combinations() {
        // Grid with 3 values → should evaluate 3 param sets.
        let factory = |params: &ParamSet| {
            let val = match params.get("constant") {
                Some(ParamValue::Float(v)) => *v,
                _ => 0.0,
            };
            Pipeline::new().estimator_step("est", Box::new(ConstantEstimator { value: val }))
        };

        let grid = crate::param_grid! {
            "constant" => [0.0_f64, 1.0_f64, 2.0_f64],
        };
        let mut gs = GridSearchCV::new(Box::new(factory), grid, Box::new(KFold::new(3)), neg_mse);

        let x = Array2::<f64>::zeros((30, 2));
        let y = Array1::<f64>::from_elem(30, 1.0);
        gs.fit(&x, &y).unwrap();

        let results = gs.cv_results().unwrap();
        assert_eq!(results.params.len(), 3);
        assert_eq!(results.mean_scores.len(), 3);
        assert_eq!(results.all_scores.len(), 3);
    }

    #[test]
    fn test_grid_search_finds_best_params() {
        // y is constant at 1.0. The estimator that predicts 1.0 should win.
        let factory = |params: &ParamSet| {
            let val = match params.get("constant") {
                Some(ParamValue::Float(v)) => *v,
                _ => 0.0,
            };
            Pipeline::new().estimator_step("est", Box::new(ConstantEstimator { value: val }))
        };

        let grid = crate::param_grid! {
            "constant" => [0.0_f64, 1.0_f64, 5.0_f64],
        };
        let mut gs = GridSearchCV::new(Box::new(factory), grid, Box::new(KFold::new(3)), neg_mse);

        let x = Array2::<f64>::zeros((30, 2));
        let y = Array1::<f64>::from_elem(30, 1.0);
        gs.fit(&x, &y).unwrap();

        let best = gs.best_params().unwrap();
        assert_eq!(best.get("constant"), Some(&ParamValue::Float(1.0)));
    }

    #[test]
    fn test_grid_search_best_score_is_zero_for_perfect() {
        let factory =
            |_params: &ParamSet| Pipeline::new().estimator_step("mean", Box::new(MeanEstimator));

        let grid = crate::param_grid! {
            "dummy" => [true],
        };
        let mut gs = GridSearchCV::new(Box::new(factory), grid, Box::new(KFold::new(5)), neg_mse);

        let x = Array2::<f64>::zeros((30, 2));
        let y = Array1::<f64>::from_elem(30, 3.0);
        gs.fit(&x, &y).unwrap();

        // Mean predictor on constant y → MSE = 0 → neg_mse = 0.
        let score = gs.best_score().unwrap();
        assert!(score.abs() < 1e-10, "expected score ~0, got {score}");
    }

    #[test]
    fn test_grid_search_empty_grid_returns_error() {
        let factory = |_: &ParamSet| Pipeline::new().estimator_step("m", Box::new(MeanEstimator));
        let mut gs = GridSearchCV::new(Box::new(factory), vec![], Box::new(KFold::new(3)), neg_mse);
        let x = Array2::<f64>::zeros((30, 2));
        let y = Array1::<f64>::zeros(30);
        assert!(gs.fit(&x, &y).is_err());
    }

    #[test]
    fn test_grid_search_returns_none_before_fit() {
        let factory = |_: &ParamSet| Pipeline::new().estimator_step("m", Box::new(MeanEstimator));
        let grid = crate::param_grid! { "dummy" => [true] };
        let gs = GridSearchCV::new(Box::new(factory), grid, Box::new(KFold::new(3)), neg_mse);
        assert!(gs.best_params().is_none());
        assert!(gs.best_score().is_none());
        assert!(gs.cv_results().is_none());
    }

    #[test]
    fn test_cv_results_structure() {
        let factory = |params: &ParamSet| {
            let val = match params.get("c") {
                Some(ParamValue::Float(v)) => *v,
                _ => 0.0,
            };
            Pipeline::new().estimator_step("est", Box::new(ConstantEstimator { value: val }))
        };
        let grid = crate::param_grid! { "c" => [1.0_f64, 2.0_f64] };
        let mut gs = GridSearchCV::new(Box::new(factory), grid, Box::new(KFold::new(4)), neg_mse);
        let x = Array2::<f64>::zeros((20, 1));
        let y = Array1::<f64>::zeros(20);
        gs.fit(&x, &y).unwrap();

        let results = gs.cv_results().unwrap();
        // 2 param sets.
        assert_eq!(results.params.len(), 2);
        // Each all_scores entry has 4 fold scores.
        for fold_scores in &results.all_scores {
            assert_eq!(fold_scores.len(), 4);
        }
    }
}
