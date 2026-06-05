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
    /// All per-fold scores for each parameter set.
    pub all_scores: Vec<Array1<f64>>,
}

impl CvResults {
    pub(crate) fn new() -> Self {
        Self {
            params: Vec::new(),
            mean_scores: Vec::new(),
            all_scores: Vec::new(),
        }
    }

    pub(crate) fn push(&mut self, params: ParamSet, scores: Array1<f64>) {
        let mean = scores.mean().unwrap_or(f64::NEG_INFINITY);
        self.params.push(params);
        self.mean_scores.push(mean);
        self.all_scores.push(scores);
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
