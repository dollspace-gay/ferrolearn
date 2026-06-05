//! Construction helpers that mirror scikit-learn's `make_pipeline` and
//! `make_union` shorthand functions.
//!
//! These auto-generate step names of the form `step0`, `step1`, ... so that
//! callers don't have to invent meaningful names when they're not needed.
//!
//! ## REQ status
//!
//! Mirrors `sklearn.pipeline.make_pipeline` (`sklearn/pipeline.py:1220`),
//! `make_union` (`:1889`), `_name_estimators` (`:1196`) at v1.5.2. Every REQ is
//! BINARY (R-DEFER-2): SHIPPED or NOT-STARTED (with a concrete blocker).
//!
//! | REQ | Status | Notes |
//! |---|---|---|
//! | REQ-MAKE-PIPELINE-MECHANIC (build a Pipeline from steps + final estimator) | SHIPPED | `make_pipeline(Vec<steps>, Option<estimator>)` assembles a `Pipeline` preserving input order; N steps + estimator. R-DEV-7 explicit `(Vec, Option)` split vs sklearn variadic `*steps` with last-is-estimator inference. Guards `step_names().len()==3`, single-step, input-order (non-commuting transforms), fit/predict. |
//! | REQ-MAKE-UNION-MECHANIC (build a FeatureUnion from transformers) | SHIPPED | `make_union(Vec<transformers>)` ⇒ `n_transformers()==N`; mirrors `make_union` (`:1889`). Guard `n_transformers()==2`. (`n_jobs`/`verbose` absent — REQ-MEMORY-VERBOSE-NJOBS.) |
//! | REQ-STEP-NAMING (class-name + `-N` dedup names) | NOT-STARTED | sklearn `_name_estimators` names steps by lowercased class name with `-N` dedup (`:1196`; `make_pipeline(SS, SS, LogReg)` → `standardscaler-1/standardscaler-2/logisticregression`); ferrolearn uses positional `step{i}`/`"estimator"`/`fu{i}`. ARCHITECTURAL: `Box<dyn PipelineStep>` erases the concrete type and the core trait exposes no type-name channel (`std::any` on a boxed `dyn` yields the trait type, not the class), so the fix needs a `PipelineStep::type_name` method in `ferrolearn-core` first. Blocker #1871. |
//! | REQ-MEMORY-VERBOSE-NJOBS (make_pipeline memory/verbose, make_union n_jobs/verbose) | NOT-STARTED | absent; multi-file (helper signatures + core `Pipeline`/`FeatureUnion` constructors). Blocker #1872. |
//! | REQ-X-1 (R-SUBSTRATE) | SHIPPED | pure construction over `num_traits::Float` + core `Pipeline` types; no direct `ndarray` array layer — R-SUBSTRATE N/A for this unit. |
//! | REQ-X-2 (non-test production consumer) | SHIPPED | re-exported `pub use helpers::{make_pipeline, make_union}` in `lib.rs` (boundary API, S5/R-DEFER-1). |

use ferrolearn_core::pipeline::{Pipeline, PipelineEstimator, PipelineStep, PipelineTransformer};
use num_traits::Float;

use crate::feature_union::FeatureUnion;

/// Build a [`Pipeline`] from a list of pre-boxed `PipelineStep`s and an
/// optional final [`PipelineEstimator`].
///
/// Auto-generates step names of the form `step0`, `step1`, ...
///
/// # Example
///
/// ```rust,no_run
/// use ferrolearn_model_sel::helpers::make_pipeline;
/// use ferrolearn_core::pipeline::Pipeline;
///
/// // In real code, `steps` would be filled with concrete transformers and
/// // `estimator` with the final fit step.
/// let _pipe: Pipeline<f64> = make_pipeline::<f64>(Vec::new(), None);
/// ```
#[must_use]
pub fn make_pipeline<F>(
    steps: Vec<Box<dyn PipelineStep<F>>>,
    estimator: Option<Box<dyn PipelineEstimator<F>>>,
) -> Pipeline<F>
where
    F: Float + Send + Sync + 'static,
{
    let mut pipe = Pipeline::<F>::new();
    for (i, step) in steps.into_iter().enumerate() {
        let name = format!("step{i}");
        pipe = pipe.step(&name, step);
    }
    if let Some(est) = estimator {
        pipe = pipe.estimator_step("estimator", est);
    }
    pipe
}

/// Build a [`FeatureUnion`] from a list of pre-boxed transformers.
///
/// Auto-generates names of the form `fu0`, `fu1`, ...
#[must_use]
pub fn make_union<F>(transformers: Vec<Box<dyn PipelineTransformer<F>>>) -> FeatureUnion<F>
where
    F: Float + Send + Sync + 'static,
{
    let mut fu = FeatureUnion::<F>::new();
    for (i, t) in transformers.into_iter().enumerate() {
        let name = format!("fu{i}");
        fu = fu.add(&name, t);
    }
    fu
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn make_pipeline_empty_runs() {
        // Just confirm the empty case constructs without panic.
        let _pipe: Pipeline<f64> = make_pipeline::<f64>(Vec::new(), None);
    }

    #[test]
    fn make_union_empty() {
        let fu: FeatureUnion<f64> = make_union::<f64>(Vec::new());
        assert_eq!(fu.n_transformers(), 0);
    }
}
