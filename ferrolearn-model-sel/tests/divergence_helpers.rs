//! Adversarial divergence audit of `ferrolearn-model-sel::helpers`
//! (`make_pipeline`, `make_union`) against scikit-learn 1.5.2
//! (`sklearn.pipeline.make_pipeline` `sklearn/pipeline.py:1220`,
//! `make_union` `:1889`, `_name_estimators` `:1196`).
//!
//! Tracking: construction mechanic SHIPPED; step-NAMING divergence is
//! ARCHITECTURAL (#1871 — the core traits expose no concrete-type-name channel);
//! `memory`/`verbose`/`n_jobs` pass-through is multi-file (#1872).
//!
//! Test classes:
//! - GREEN GUARDS (must PASS): the SHIPPED construction mechanic —
//!   `make_pipeline` builds an N-transformer + 1-estimator pipeline that fits,
//!   predicts, and preserves input order; `make_union` builds a K-transformer
//!   union. Expected COUNTS come from a LIVE sklearn 1.5.2 oracle run from /tmp
//!   (R-CHAR-3 — never copied from the ferrolearn side).
//!
//! Live oracle values pinned below (recomputed 2026-06, sklearn 1.5.2):
//!   make_pipeline(StandardScaler(), StandardScaler(), LogisticRegression())
//!     -> len(.steps) == 3      (SK_PIPE_3STEP_COUNT)
//!   make_pipeline(StandardScaler())
//!     -> len(.steps) == 1      (SK_PIPE_1STEP_COUNT)
//!   make_pipeline()  (zero args)
//!     -> len(.steps) == 0, NO error (SK_PIPE_0STEP_COUNT)
//!   make_union(PCA(), TruncatedSVD())
//!     -> len(.transformer_list) == 2  (SK_UNION_2_COUNT)
//!
//! The step-NAMING divergence (sklearn `['standardscaler-1','standardscaler-2',
//! 'logisticregression']` vs ferrolearn `['step0','step1','estimator']`) is
//! deliberately NOT pinned here: it is not fixable inside `helpers.rs` alone
//! (the boxed `dyn PipelineStep<F>` has type-erased its concrete class; the core
//! traits declare no `type_name()` and do not extend `Any`, so neither
//! `std::any::type_name::<T>()` — `T` unavailable at the call site — nor
//! `std::any::type_name_of_val(&*boxed)` — yields `dyn PipelineStep`, not the
//! estimator class — can recover the class name). Tracked architectural: #1871.

use ferrolearn_core::error::FerroError;
use ferrolearn_core::pipeline::{
    FittedPipelineEstimator, FittedPipelineTransformer, Pipeline, PipelineEstimator, PipelineStep,
    PipelineTransformer, as_estimator_step, as_transform_step,
};
use ferrolearn_core::{Fit, Predict};
use ferrolearn_model_sel::{make_pipeline, make_union};
use ndarray::{Array1, Array2, array};

// --- Live-oracle constants (sklearn 1.5.2, computed from /tmp) ---------------
// R-CHAR-3: these are the SKLEARN side, never copied from ferrolearn.
const SK_PIPE_3STEP_COUNT: usize = 3; // len(make_pipeline(SS, SS, LogReg).steps)
const SK_PIPE_1STEP_COUNT: usize = 1; // len(make_pipeline(SS).steps)
const SK_UNION_2_COUNT: usize = 2; // len(make_union(PCA, TruncatedSVD).transformer_list)

// ---------------------------------------------------------------------------
// Order-distinguishable fixtures. `AddK` and `MulK` do NOT commute, so the
// fitted output value witnesses the EXECUTION ORDER of the steps.
// ---------------------------------------------------------------------------

/// Adds a constant `k` to every element.
struct AddK(f64);
impl PipelineTransformer<f64> for AddK {
    fn fit_pipeline(
        &self,
        _x: &Array2<f64>,
        _y: &Array1<f64>,
    ) -> Result<Box<dyn FittedPipelineTransformer<f64>>, FerroError> {
        Ok(Box::new(FittedAddK(self.0)))
    }
}
struct FittedAddK(f64);
impl FittedPipelineTransformer<f64> for FittedAddK {
    fn transform_pipeline(&self, x: &Array2<f64>) -> Result<Array2<f64>, FerroError> {
        Ok(x.mapv(|v| v + self.0))
    }
}

/// Multiplies every element by a constant `k`.
struct MulK(f64);
impl PipelineTransformer<f64> for MulK {
    fn fit_pipeline(
        &self,
        _x: &Array2<f64>,
        _y: &Array1<f64>,
    ) -> Result<Box<dyn FittedPipelineTransformer<f64>>, FerroError> {
        Ok(Box::new(FittedMulK(self.0)))
    }
}
struct FittedMulK(f64);
impl FittedPipelineTransformer<f64> for FittedMulK {
    fn transform_pipeline(&self, x: &Array2<f64>) -> Result<Array2<f64>, FerroError> {
        Ok(x.mapv(|v| v * self.0))
    }
}

/// Estimator that sums each row.
struct RowSum;
impl PipelineEstimator<f64> for RowSum {
    fn fit_pipeline(
        &self,
        _x: &Array2<f64>,
        _y: &Array1<f64>,
    ) -> Result<Box<dyn FittedPipelineEstimator<f64>>, FerroError> {
        Ok(Box::new(FittedRowSum))
    }
}
struct FittedRowSum;
impl FittedPipelineEstimator<f64> for FittedRowSum {
    fn predict_pipeline(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
        Ok(Array1::from_vec(
            x.rows().into_iter().map(|r| r.sum()).collect(),
        ))
    }
}

/// Identity transformer (for the union count guard).
struct Identity;
impl PipelineTransformer<f64> for Identity {
    fn fit_pipeline(
        &self,
        _x: &Array2<f64>,
        _y: &Array1<f64>,
    ) -> Result<Box<dyn FittedPipelineTransformer<f64>>, FerroError> {
        Ok(Box::new(FittedIdentity))
    }
}
struct FittedIdentity;
impl FittedPipelineTransformer<f64> for FittedIdentity {
    fn transform_pipeline(&self, x: &Array2<f64>) -> Result<Array2<f64>, FerroError> {
        Ok(x.clone())
    }
}

// ---------------------------------------------------------------------------
// GREEN GUARD: REQ-MAKE-PIPELINE-MECHANIC — non-empty step COUNT parity.
// Oracle: len(make_pipeline(StandardScaler(), StandardScaler(),
//             LogisticRegression()).steps) == 3.
// The unfitted `Pipeline` exposes no public count accessor, so the count is
// observed via `FittedPipeline::step_names()` after a successful fit.
// ---------------------------------------------------------------------------
#[test]
fn green_make_pipeline_two_transformers_one_estimator_count() {
    let steps: Vec<Box<dyn PipelineStep<f64>>> =
        vec![as_transform_step(AddK(0.0)), as_transform_step(MulK(1.0))];
    let pipe: Pipeline<f64> = make_pipeline(steps, Some(Box::new(RowSum)));

    let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let y = array![0.0, 1.0];
    let fitted = pipe.fit(&x, &y).expect("pipeline must fit");

    // 2 transformer steps + 1 estimator step == 3 named steps.
    assert_eq!(
        fitted.step_names().len(),
        SK_PIPE_3STEP_COUNT,
        "make_pipeline(2 transformers, estimator) must produce {SK_PIPE_3STEP_COUNT} steps \
         (sklearn make_pipeline(SS, SS, LogReg).steps len)"
    );
}

// ---------------------------------------------------------------------------
// GREEN GUARD: REQ-MAKE-PIPELINE-MECHANIC — single step.
// Oracle: len(make_pipeline(StandardScaler()).steps) == 1.
// ferrolearn single transformer + estimator slot empty cannot be fit (no final
// estimator, owned by #361); express the 1-step analog as a single ESTIMATOR.
// ---------------------------------------------------------------------------
#[test]
fn green_make_pipeline_single_step_count() {
    let pipe: Pipeline<f64> = make_pipeline(Vec::new(), Some(Box::new(RowSum)));
    let x = array![[1.0, 2.0]];
    let y = array![0.0];
    let fitted = pipe
        .fit(&x, &y)
        .expect("single-estimator pipeline must fit");
    assert_eq!(
        fitted.step_names().len(),
        SK_PIPE_1STEP_COUNT,
        "single-step make_pipeline must produce {SK_PIPE_1STEP_COUNT} step"
    );
}

// ---------------------------------------------------------------------------
// GREEN GUARD: REQ-MAKE-PIPELINE-MECHANIC — STEP ORDER preserved + end-to-end
// fit/predict. AddK(10) then MulK(2) is order-sensitive: applied in input order
// to [1,2,3] -> ((x+10)*2) = [22,24,26], row sum = 72. The reversed order would
// give (x*2+10) = [12,14,16], sum 42 — so a passing 72 witnesses input order.
// ---------------------------------------------------------------------------
#[test]
fn green_make_pipeline_order_and_fit_predict() {
    let steps: Vec<Box<dyn PipelineStep<f64>>> =
        vec![as_transform_step(AddK(10.0)), as_transform_step(MulK(2.0))];
    let pipe: Pipeline<f64> = make_pipeline(steps, Some(Box::new(RowSum)));

    let x = array![[1.0, 2.0, 3.0]];
    let y = array![0.0];
    let fitted = pipe.fit(&x, &y).expect("pipeline must fit");
    let preds = fitted.predict(&x).expect("pipeline must predict");

    // Step names preserve input order: step0, step1, estimator.
    assert_eq!(fitted.step_names(), vec!["step0", "step1", "estimator"]);

    // ((1+10)*2) + ((2+10)*2) + ((3+10)*2) = 22 + 24 + 26 = 72.
    assert_eq!(preds.len(), 1);
    assert!(
        (preds[0] - 72.0).abs() < 1e-12,
        "input-order execution must yield 72.0, got {} (reversed order would be 42.0)",
        preds[0]
    );
}

// ---------------------------------------------------------------------------
// GREEN GUARD: REQ-MAKE-PIPELINE-MECHANIC — empty case is COMPATIBLE, not a
// divergence. sklearn make_pipeline() with ZERO args returns a Pipeline with 0
// steps and does NOT raise (verified live, sklearn 1.5.2):
//   make_pipeline()  ->  len(.steps) == 0, no error.
// ferrolearn make_pipeline(Vec::new(), None) likewise constructs without panic.
// (The empty pipeline cannot be fit — no final estimator, owned by #361 — which
// is consistent: sklearn's empty pipeline also cannot .predict.)
// ---------------------------------------------------------------------------
#[test]
fn green_make_pipeline_empty_constructs() {
    let pipe: Pipeline<f64> = make_pipeline(Vec::new(), None);
    // No public count accessor on the unfitted pipeline; the contract here is
    // "constructs without panic", mirroring sklearn returning an empty Pipeline.
    // Fitting it must error (no estimator) — same observable as sklearn's empty
    // pipeline having no final predictor.
    let x = array![[1.0]];
    let y = array![0.0];
    assert!(
        pipe.fit(&x, &y).is_err(),
        "empty pipeline must reject fit (no final estimator)"
    );
}

// ---------------------------------------------------------------------------
// GREEN GUARD: REQ-MAKE-UNION-MECHANIC — transformer COUNT parity.
// Oracle: len(make_union(PCA(), TruncatedSVD()).transformer_list) == 2.
// ---------------------------------------------------------------------------
#[test]
fn green_make_union_two_transformers_count() {
    let transformers: Vec<Box<dyn PipelineTransformer<f64>>> =
        vec![Box::new(Identity), Box::new(MulK(2.0))];
    let fu = make_union(transformers);
    assert_eq!(
        fu.n_transformers(),
        SK_UNION_2_COUNT,
        "make_union(2 transformers).n_transformers() must equal {SK_UNION_2_COUNT} \
         (sklearn make_union(PCA, TruncatedSVD).transformer_list len)"
    );
    // Order preserved: fu0 then fu1.
    assert_eq!(fu.transformer_names(), vec!["fu0", "fu1"]);
}

// ---------------------------------------------------------------------------
// GREEN GUARD: REQ-X-2 — `make_pipeline`/`make_union` are crate public API via
// the `lib.rs` re-export. This file imports them as
// `ferrolearn_model_sel::{make_pipeline, make_union}` (not via the `helpers`
// submodule path), which compiles ONLY if the re-export exists.
// ---------------------------------------------------------------------------
#[test]
fn green_reexport_is_public_api() {
    // Build via the re-exported free fns and the `.step()`-style wrapper to
    // exercise the PipelineStep path used by make_pipeline.
    let _est_step: Box<dyn PipelineStep<f64>> = as_estimator_step(RowSum);
    let _pipe: Pipeline<f64> = make_pipeline(Vec::new(), None);
    let _fu = make_union::<f64>(Vec::new());
}
