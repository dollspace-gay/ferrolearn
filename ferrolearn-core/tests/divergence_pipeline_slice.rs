//! Divergence pin: `Pipeline::into_slice` rejects out-of-bounds / inverted
//! ranges with an error, whereas sklearn's `Pipeline.__getitem__` slice arm
//! uses Python slice semantics, which CLAMP (and never raise) for an end past
//! the number of steps, a start past the number of steps, or start > end.
//!
//! ferrolearn cite: `ferrolearn-core/src/pipeline.rs:513-522`
//!   ```text
//!   pub fn into_slice(self, start: usize, end: usize) -> Result<Pipeline<F>, FerroError> {
//!       let n_steps = self.len();
//!       if start > end || end > n_steps {
//!           return Err(FerroError::InvalidParameter { ... });
//!       }
//!   ```
//!
//! sklearn cite: `sklearn/pipeline.py:307-312`
//!   ```text
//!   if isinstance(ind, slice):
//!       if ind.step not in (1, None):
//!           raise ValueError("Pipeline slicing only supports a step of 1")
//!       return self.__class__(
//!           self.steps[ind], memory=self.memory, verbose=self.verbose
//!       )
//!   ```
//!   `self.steps[ind]` is an ordinary Python list slice, so it CLAMPS rather
//!   than raising for out-of-range bounds.
//!
//! Live oracle (sklearn 1.5.2), on a 3-step pipeline
//!   `Pipeline([('a',StandardScaler()),('b',MinMaxScaler()),('c',GaussianNB())])`:
//!   ```text
//!   p[0:100].steps -> ['a','b','c']   # end past len: CLAMP to all steps, NO error
//!   p[5:100].steps -> []              # start past len: EMPTY, NO error
//!   p[2:1].steps   -> []              # start > end: EMPTY, NO error
//!   ```
//!   ferrolearn `into_slice` returns `Err(InvalidParameter)` for ALL three.
//!   This is a SEMANTIC divergence (erroring where Python clamps / returns
//!   empty), not a Rust-idiom shape difference (Option-for-None is acceptable
//!   under R-CODE-2; raising-vs-clamping is not — the observable result differs:
//!   sklearn produces a valid sub-pipeline, ferrolearn produces no pipeline).
//!
//! Tracking: #2235

use ferrolearn_core::error::FerroError;
use ferrolearn_core::pipeline::{
    FittedPipelineEstimator, FittedPipelineTransformer, Pipeline, PipelineEstimator,
    PipelineTransformer,
};
use ndarray::{Array1, Array2};

struct IdT;
impl PipelineTransformer<f64> for IdT {
    fn fit_pipeline(
        &self,
        _x: &Array2<f64>,
        _y: &Array1<f64>,
    ) -> Result<Box<dyn FittedPipelineTransformer<f64>>, FerroError> {
        Ok(Box::new(FIdT))
    }
}
struct FIdT;
impl FittedPipelineTransformer<f64> for FIdT {
    fn transform_pipeline(&self, x: &Array2<f64>) -> Result<Array2<f64>, FerroError> {
        Ok(x.clone())
    }
}

struct SumE;
impl PipelineEstimator<f64> for SumE {
    fn fit_pipeline(
        &self,
        _x: &Array2<f64>,
        _y: &Array1<f64>,
    ) -> Result<Box<dyn FittedPipelineEstimator<f64>>, FerroError> {
        Ok(Box::new(FSumE))
    }
}
struct FSumE;
impl FittedPipelineEstimator<f64> for FSumE {
    fn predict_pipeline(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
        Ok(Array1::from_iter(x.rows().into_iter().map(|r| r.sum())))
    }
}

fn build_3step() -> Pipeline<f64> {
    Pipeline::new()
        .transform_step("a", Box::new(IdT))
        .transform_step("b", Box::new(IdT))
        .estimator_step("c", Box::new(SumE))
}

/// Divergence: `Pipeline::into_slice` raises where sklearn's slice CLAMPS.
///
/// sklearn (`pipeline.py:307-312`, list-slice semantics) returns a valid
/// sub-pipeline for an out-of-range / inverted slice; ferrolearn
/// (`pipeline.rs:515`) returns `Err`. Tracking: #2235.
#[test]
#[ignore = "divergence: into_slice errors on OOB/inverted ranges where sklearn list-slice clamps to a valid sub-pipeline; tracking #2235"]
fn divergence_into_slice_clamps_like_python() {
    // Oracle: p[0:100] on 3 steps -> ['a','b','c'] (end past len: clamp to all,
    // NO error). ferrolearn errors because end > n_steps.
    let sub = build_3step()
        .into_slice(0, 100)
        .expect("sklearn p[0:100] clamps to all 3 steps, NO error; ferrolearn errors");
    assert_eq!(sub.step_names(), vec!["a", "b", "c"]);

    // Oracle: p[5:100] on 3 steps -> [] (start past len: empty, NO error).
    // ferrolearn errors because end > n_steps (and start > n_steps).
    let sub = build_3step()
        .into_slice(5, 100)
        .expect("sklearn p[5:100] (start past len) -> empty pipeline, NO error; ferrolearn errors");
    assert!(sub.step_names().is_empty());

    // Oracle: p[2:1] on 3 steps -> [] (start > end: empty, NO error).
    // ferrolearn errors because start > end.
    let sub = build_3step()
        .into_slice(2, 1)
        .expect("sklearn p[2:1] (start>end) -> empty pipeline, NO error; ferrolearn errors");
    assert!(sub.step_names().is_empty());
}
