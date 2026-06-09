//! Divergence: `RegressorChain`/`ClassifierChain` accept a non-permutation
//! `order` of the correct LENGTH; scikit-learn 1.5.2 rejects it.
//!
//! sklearn `_BaseChain.fit` (`sklearn/multioutput.py:733`):
//!   `elif sorted(self.order_) != list(range(Y.shape[1])):`
//!   `    raise ValueError("invalid order")`
//! i.e. `order` must be a PERMUTATION of `range(n_targets)`, not merely the
//! right length. `order=[0, 0]` (length 2, n_targets 2, but not a permutation)
//! raises `ValueError("invalid order")` (verified live, sklearn 1.5.2).
//!
//! ferrolearn `fit_chain` (`ferrolearn-model-sel/src/chain.rs:273-287`) checks
//! ONLY `o.len() != n_targets`; `[0, 0]` passes and the chain silently fits
//! target 0 twice and NEVER fits target 1 -> a wrong fitted model with no error
//! (R-DEV-2 default/exception parity violation + R-DEV-3 wrong output column 1).
//!
//! Tracking: #2380
//!
//! LIVE ORACLE (R-CHAR-3):
//!   import numpy as np
//!   from sklearn.multioutput import RegressorChain
//!   from sklearn.linear_model import Ridge
//!   X = np.array([[1.,0.],[0.,1.],[1.,1.],[2.,1.]])
//!   Y = np.column_stack([X[:,0]+1.0, X[:,1]+2.0])
//!   RegressorChain(Ridge(alpha=1.0), order=[0,0]).fit(X,Y)
//!   # -> ValueError: invalid order

use ferrolearn_core::pipeline::{FittedPipelineEstimator, Pipeline, PipelineEstimator};
use ferrolearn_core::FerroError;
use ferrolearn_model_sel::chain::RegressorChain;
use ndarray::{Array1, Array2};

type PipelineFactory = Box<dyn Fn() -> Pipeline<f64> + Send + Sync>;

struct MeanBase;
struct FittedMean(f64);
impl PipelineEstimator<f64> for MeanBase {
    fn fit_pipeline(
        &self,
        _x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<Box<dyn FittedPipelineEstimator<f64>>, FerroError> {
        Ok(Box::new(FittedMean(y.iter().copied().sum::<f64>() / y.len() as f64)))
    }
}
impl FittedPipelineEstimator<f64> for FittedMean {
    fn predict_pipeline(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
        Ok(Array1::from_elem(x.nrows(), self.0))
    }
}
fn mean_factory() -> PipelineFactory {
    Box::new(|| Pipeline::<f64>::new().estimator_step("mean", Box::new(MeanBase)))
}

#[test]
#[ignore = "divergence: non-permutation order accepted (only length checked); sklearn ValueError('invalid order') multioutput.py:733; tracking #2380"]
fn divergence_chain_order_must_be_permutation() {
    let x = ndarray::array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]];
    let y = ndarray::array![[2.0, 2.0], [1.0, 3.0], [2.0, 3.0], [3.0, 3.0]]; // 2 targets

    // order=[0,0]: correct length (2) but NOT a permutation of {0,1}.
    // sklearn raises ValueError("invalid order"); ferrolearn must reject too.
    let result = RegressorChain::new(mean_factory()).order(vec![0, 0]).fit(&x, &y);
    match result {
        Err(_) => {}
        Ok(_) => panic!(
            "order=[0,0] accepted: sklearn raises ValueError('invalid order') \
             (multioutput.py:733); ferrolearn only checks length and silently \
             fits target 0 twice"
        ),
    }
}
