//! Divergence pin for `OutputCodeClassifier` degenerate floor-to-zero
//! (`ferrolearn-model-sel/src/chain.rs`) vs scikit-learn 1.5.2
//! `sklearn/multiclass.py` `OutputCodeClassifier.fit`.
//!
//! ACToR-critic deliverable. Newly reachable after commit 41b3fc24 changed
//! `n_codes` from `ceil(code_size*k).max(2)` (always >= 2) to
//! `(code_size*k) as usize` (multiclass.py:1189, `int(n_classes*code_size)`).
//! When `int(n_classes*code_size) == 0` the floor-to-zero path is now
//! reachable, and the two libraries DISAGREE on whether `fit` succeeds.
//!
//! LIVE ORACLE (R-CHAR-3 — sklearn 1.5.2):
//!   python3 -c "
//!   import numpy as np
//!   from sklearn.multiclass import OutputCodeClassifier
//!   from sklearn.tree import DecisionTreeClassifier
//!   rng=np.random.RandomState(0); X=rng.rand(40,3)
//!   y=np.array([i%3 for i in range(40)])   # k=3 distinct classes
//!   OutputCodeClassifier(DecisionTreeClassifier(random_state=0),
//!                        code_size=0.1, random_state=0).fit(X, y)"
//!   # int(3 * 0.1) == int(0.30000000000000004) == 0
//!   # -> raises IndexError: list index out of range  (multiclass.py:1215,
//!   #    `self.estimators_[0].n_features_in_` indexes the empty estimators_)
//!   # Deterministic for every (cs,k) with int(cs*k)==0 (e.g. 0.01/5, 0.2/4).
//!
//! ferrolearn (chain.rs:381 `let n_codes = (self.code_size * k as f64) as usize`):
//!   code_size=0.1, k=3 => n_codes == 0; `fit` SUCCEEDS returning a
//!   FittedOutputCodeClassifier with n_estimators()==0, and `predict` then
//!   returns all class-0 labels (argmin over zero-width distance picks index 0).
//!   sklearn raises at fit; ferrolearn fits + predicts successfully.

use ferrolearn_core::FerroError;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, Pipeline, PipelineEstimator};
use ferrolearn_model_sel::chain::OutputCodeClassifier;
use ndarray::{Array1, Array2};

type PipelineFactory = Box<dyn Fn() -> Pipeline<f64> + Send + Sync>;

struct MeanEstimator;
struct FittedMean(f64);
impl PipelineEstimator<f64> for MeanEstimator {
    fn fit_pipeline(
        &self,
        _x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<Box<dyn FittedPipelineEstimator<f64>>, FerroError> {
        Ok(Box::new(FittedMean(
            y.iter().copied().sum::<f64>() / y.len() as f64,
        )))
    }
}
impl FittedPipelineEstimator<f64> for FittedMean {
    fn predict_pipeline(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
        Ok(Array1::from_elem(x.nrows(), self.0))
    }
}
fn mean_factory() -> PipelineFactory {
    Box::new(|| Pipeline::<f64>::new().estimator_step("mean", Box::new(MeanEstimator)))
}

// #1839 new-blocker
//
// sklearn `OutputCodeClassifier(..., code_size=0.1).fit(X, y)` with k=3
// classes RAISES (IndexError, multiclass.py:1215) because
// int(3*0.1)==0 => empty estimators_. ferrolearn's fit SUCCEEDS with
// n_estimators()==0. This assertion (fit must be Err, mirroring the sklearn
// raise) FAILS against the current implementation: ferrolearn returns Ok.
#[test]
#[ignore = "divergence: OCC fit succeeds with n_codes==0 where sklearn raises (multiclass.py:1215); tracking #1839"]
fn divergence_occ_degenerate_zero_codes_should_error() {
    let x = Array2::<f64>::zeros((9, 2));
    let y = Array1::from(vec![0usize, 0, 0, 1, 1, 1, 2, 2, 2]); // k = 3
    // code_size=0.1 => int(0.1*3)=int(0.30000000000000004)=0 estimators.
    let result = OutputCodeClassifier::new(mean_factory())
        .code_size(0.1)
        .random_state(0)
        .fit(&x, &y);
    // sklearn raises at fit for the floor-to-zero case (multiclass.py:1215).
    assert!(
        result.is_err(),
        "code_size=0.1,k=3 floors to 0 codes; sklearn raises IndexError at fit \
         (multiclass.py:1215), but ferrolearn fits successfully with \
         n_estimators()==0 and predicts all class 0"
    );
}
