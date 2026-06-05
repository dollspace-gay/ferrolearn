//! Divergence / conformance audit for `ferrolearn-model-sel/src/chain.rs`
//! against scikit-learn 1.5.2 `ClassifierChain` / `RegressorChain`
//! (`sklearn/multioutput.py` `_BaseChain` :700-831, ClassifierChain :832,
//! RegressorChain :1106) and `OutputCodeClassifier`
//! (`sklearn/multiclass.py:1025`, `fit` :1170-1220, `predict` :1222-1245).
//!
//! ACToR-critic deliverable. Every expected value is grounded in a LIVE
//! sklearn 1.5.2 oracle call (R-CHAR-3 — NEVER copied from the ferrolearn
//! side). Oracle commands are quoted next to each assertion.
//!
//! - GREEN guards lock in the SHIPPED claims (REQ-CHAIN-CORE for both chains,
//!   end-to-end + non-identity-order un-permute; REQ-CHAIN-ORDER; REQ-OCC-MECH).
//! - The `#[ignore]`d `divergence_occ_*` pins are the DETERMINISTIC FIXABLE
//!   divergence #1830 (`n_codes` formula) and its coupled `code_size<=0`
//!   validation facet. They FAIL un-ignored against the current implementation.

use ferrolearn_core::FerroError;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, Pipeline, PipelineEstimator};
use ferrolearn_core::traits::Predict;
use ferrolearn_model_sel::chain::{ClassifierChain, OutputCodeClassifier, RegressorChain};
use ndarray::{Array1, Array2};

type PipelineFactory = Box<dyn Fn() -> Pipeline<f64> + Send + Sync>;

// ---------------------------------------------------------------------------
// Deterministic 1-nearest-neighbour base estimator.
//
// At fit it memorises the (augmented-feature row -> y) training pairs; at
// predict it returns the y of the closest stored row (squared-euclidean,
// first-on-tie). For the chain tests below the predict-time X equals the fit
// X and the cascaded prior features are EXACT (because the base reproduces
// the training target exactly), so every predict row matches a stored row
// exactly and the chain reproduces Y element-for-element — exactly the
// `np.allclose(P, Y) == True` the live oracle reports for a perfectly-fitting
// base. This makes the chain CORE + UN-PERMUTE checkable end-to-end without a
// random base (RNG carve-out avoided).
// ---------------------------------------------------------------------------
struct NnEstimator;
struct FittedNn {
    rows: Vec<Vec<f64>>,
    targets: Vec<f64>,
}

impl PipelineEstimator<f64> for NnEstimator {
    fn fit_pipeline(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<Box<dyn FittedPipelineEstimator<f64>>, FerroError> {
        let rows: Vec<Vec<f64>> = (0..x.nrows()).map(|i| x.row(i).to_vec()).collect();
        Ok(Box::new(FittedNn {
            rows,
            targets: y.to_vec(),
        }))
    }
}

impl FittedPipelineEstimator<f64> for FittedNn {
    fn predict_pipeline(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
        let mut out = Array1::<f64>::zeros(x.nrows());
        for i in 0..x.nrows() {
            let row = x.row(i);
            let mut best = 0usize;
            let mut best_d = f64::INFINITY;
            for (k, stored) in self.rows.iter().enumerate() {
                let mut d = 0.0_f64;
                for (a, b) in row.iter().zip(stored.iter()) {
                    let diff = a - b;
                    d += diff * diff;
                }
                if d < best_d {
                    best_d = d;
                    best = k;
                }
            }
            out[i] = self.targets[best];
        }
        Ok(out)
    }
}

fn nn_factory() -> PipelineFactory {
    Box::new(|| Pipeline::<f64>::new().estimator_step("nn", Box::new(NnEstimator)))
}

// Trivial mean estimator — enough for OutputCodeClassifier structure tests.
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

// X / Y fixture shared with the live oracle (see module-level command blocks).
fn reg_fixture() -> (Array2<f64>, Array2<f64>) {
    let x = ndarray::array![
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 1.0],
        [0.0, 2.0],
        [2.0, 2.0]
    ];
    // y0 = 2*x0 + 3*x1 + 1 ; y1 = -x0 + 4*x1  (exact, no noise).
    let mut y = Array2::<f64>::zeros((6, 2));
    for i in 0..6 {
        let (x0, x1) = (x[[i, 0]], x[[i, 1]]);
        y[[i, 0]] = 2.0 * x0 + 3.0 * x1 + 1.0;
        y[[i, 1]] = -x0 + 4.0 * x1;
    }
    (x, y)
}

fn approx_eq(a: &Array2<f64>, b: &Array2<f64>) -> bool {
    a.shape() == b.shape() && a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < 1e-9)
}

// ---------------------------------------------------------------------------
// REQ-CHAIN-CORE — RegressorChain, default order (cv=None semantics).
//
// LIVE ORACLE (R-CHAR-3):
//   cd /tmp && python3 -c "
//   import numpy as np
//   from sklearn.multioutput import RegressorChain
//   from sklearn.linear_model import LinearRegression
//   X = np.array([[1.,0.],[0.,1.],[1.,1.],[2.,1.],[0.,2.],[2.,2.]])
//   Y = np.column_stack([2*X[:,0]+3*X[:,1]+1.0, -1*X[:,0]+4*X[:,1]])
//   rc = RegressorChain(LinearRegression(), cv=None).fit(X, Y)
//   print(list(rc.order_), np.allclose(rc.predict(X), Y))"
//   # -> [0, 1] True
// ---------------------------------------------------------------------------
#[test]
fn regressor_chain_core_recovers_y_default_order() {
    let (x, y) = reg_fixture();
    let fitted = RegressorChain::new(nn_factory()).fit(&x, &y).unwrap();
    assert_eq!(fitted.order(), &[0, 1]); // default order 0..n_targets
    let p = fitted.predict(&x).unwrap();
    assert_eq!(p.dim(), (6, 2));
    // sklearn: np.allclose(P, Y) == True for a perfectly-fitting base.
    assert!(approx_eq(&p, &y), "chain did not recover Y; P = {p:?}");
}

// ---------------------------------------------------------------------------
// REQ-CHAIN-CORE + REQ-CHAIN-ORDER — RegressorChain, NON-IDENTITY order=[1,0].
// The adversarial check: output MUST be in ORIGINAL column order (un-permute),
// NOT chain order. col0 of the output must equal Y[:,0] even though target 1
// was fitted first.
//
// LIVE ORACLE (R-CHAR-3):
//   rc = RegressorChain(LinearRegression(), order=[1,0], cv=None).fit(X, Y)
//   P = rc.predict(X)
//   print(list(rc.order_), np.allclose(P, Y))   # -> [1, 0] True
//   print(np.allclose(P[:,0], Y[:,0]))          # -> True  (ORIGINAL col order)
// ---------------------------------------------------------------------------
#[test]
fn regressor_chain_core_unpermutes_non_identity_order() {
    let (x, y) = reg_fixture();
    let fitted = RegressorChain::new(nn_factory())
        .order(vec![1, 0])
        .fit(&x, &y)
        .unwrap();
    assert_eq!(fitted.order(), &[1, 0]);
    let p = fitted.predict(&x).unwrap();
    // Whole matrix must equal Y in ORIGINAL column order (oracle allclose True).
    assert!(
        approx_eq(&p, &y),
        "non-identity order not un-permuted to original columns; P = {p:?}"
    );
    // Pin the un-permute explicitly: output col 0 is target 0 (Y[:,0]),
    // NOT the first-fitted target 1. A chain-order bug would put Y[:,1] here.
    for i in 0..6 {
        assert!(
            (p[[i, 0]] - y[[i, 0]]).abs() < 1e-9,
            "output col0 row{i} = {} expected Y[:,0] = {} (chain-order leak?)",
            p[[i, 0]],
            y[[i, 0]]
        );
    }
}

// ---------------------------------------------------------------------------
// REQ-CHAIN-CORE — ClassifierChain, default + non-identity order.
//
// LIVE ORACLE (R-CHAR-3):
//   cd /tmp && python3 -c "
//   import numpy as np
//   from sklearn.multioutput import ClassifierChain
//   from sklearn.tree import DecisionTreeClassifier
//   X = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.],[2.,0.],[2.,1.]])
//   Y = np.column_stack([(X[:,0]>=1).astype(int), (X[:,1]==1).astype(int)])
//   for order in (None, [1,0]):
//       cc = ClassifierChain(DecisionTreeClassifier(random_state=0),
//                            order=order, cv=None).fit(X, Y)
//       print(order, np.allclose(cc.predict(X), Y))"
//   # -> None True ; [1, 0] True
// ---------------------------------------------------------------------------
fn clf_fixture() -> (Array2<f64>, Array2<f64>) {
    let x = ndarray::array![
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [2.0, 0.0],
        [2.0, 1.0]
    ];
    let mut y = Array2::<f64>::zeros((6, 2));
    for i in 0..6 {
        y[[i, 0]] = if x[[i, 0]] >= 1.0 { 1.0 } else { 0.0 };
        y[[i, 1]] = if x[[i, 1]] == 1.0 { 1.0 } else { 0.0 };
    }
    (x, y)
}

#[test]
fn classifier_chain_core_recovers_y_both_orders() {
    let (x, y) = clf_fixture();

    // Default order.
    let fitted = ClassifierChain::new(nn_factory()).fit(&x, &y).unwrap();
    assert_eq!(fitted.order(), &[0, 1]);
    let p = fitted.predict(&x).unwrap();
    assert!(approx_eq(&p, &y), "default-order chain mismatch; P = {p:?}");

    // Non-identity order=[1,0] -> output must still be original column order.
    let fitted2 = ClassifierChain::new(nn_factory())
        .order(vec![1, 0])
        .fit(&x, &y)
        .unwrap();
    let p2 = fitted2.predict(&x).unwrap();
    assert!(
        approx_eq(&p2, &y),
        "order=[1,0] not un-permuted to original columns; P = {p2:?}"
    );
    for i in 0..6 {
        assert!(
            (p2[[i, 0]] - y[[i, 0]]).abs() < 1e-9,
            "ClassifierChain output col0 row{i} chain-order leak"
        );
    }
}

// ---------------------------------------------------------------------------
// REQ-CHAIN-ORDER — order length must equal n_targets (else InvalidParameter);
// default order is 0..K.
// sklearn raises ValueError on a non-permutation order (multioutput.py:733);
// ferrolearn validates LENGTH only (documented underclaim, not pinned here).
// ---------------------------------------------------------------------------
#[test]
fn chain_order_length_mismatch_is_invalid_parameter() {
    let (x, y) = reg_fixture(); // 2 targets
    let result = RegressorChain::new(nn_factory())
        .order(vec![0, 1, 2]) // wrong length
        .fit(&x, &y);
    // FittedRegressorChain is not Debug, so match instead of unwrap_err.
    match result {
        Err(FerroError::InvalidParameter { .. }) => {}
        Err(other) => panic!("expected InvalidParameter, got {other:?}"),
        Ok(_) => panic!("expected InvalidParameter for order length mismatch, fit succeeded"),
    }

    // Single-target chain -> 1 estimator, shape (n, 1).
    let x1 = Array2::<f64>::zeros((4, 2));
    let y1 = ndarray::array![[1.0], [2.0], [3.0], [4.0]];
    let f = RegressorChain::new(nn_factory()).fit(&x1, &y1).unwrap();
    assert_eq!(f.n_targets(), 1);
    assert_eq!(f.predict(&x1).unwrap().dim(), (4, 1));
}

// ---------------------------------------------------------------------------
// REQ-OCC-MECH — OutputCodeClassifier structure (per-column binary fit +
// nearest-code euclidean argmin). Structure only; exact labels are the RNG
// carve-out (#1836).
//
// LIVE ORACLE (R-CHAR-3):
//   cd /tmp && python3 -c "
//   import numpy as np
//   from sklearn.multiclass import OutputCodeClassifier
//   from sklearn.ensemble import RandomForestClassifier
//   X = np.repeat(np.arange(3),3).reshape(-1,1).astype(float)
//   y = np.repeat(np.arange(3),3)
//   occ = OutputCodeClassifier(RandomForestClassifier(random_state=0),
//                              code_size=2.0, random_state=0).fit(X, y)
//   print(occ.classes_.tolist(), len(occ.estimators_),
//         set(occ.predict(X).tolist()) <= {0,1,2})"
//   # -> [0, 1, 2] 6 True
// ---------------------------------------------------------------------------
#[test]
fn output_code_mechanic_structure() {
    let x = Array2::<f64>::zeros((9, 2));
    let y = Array1::from(vec![0usize, 0, 0, 1, 1, 1, 2, 2, 2]);
    let fitted = OutputCodeClassifier::new(mean_factory())
        .code_size(2.0)
        .random_state(7)
        .fit(&x, &y)
        .unwrap();
    assert_eq!(fitted.classes(), &[0, 1, 2]); // sorted, deduped
    assert!(fitted.n_estimators() > 0);
    let preds = fitted.predict(&x).unwrap();
    assert_eq!(preds.len(), 9);
    for &p in preds.iter() {
        assert!([0, 1, 2].contains(&p), "predicted label {p} not in classes");
    }

    // < 2 distinct classes -> error (ferrolearn rejects; documented stricter
    // guard, asserted here only as the current contract).
    let x3 = x.slice(ndarray::s![0..3, ..]).to_owned();
    let y1 = Array1::from(vec![0usize, 0, 0]);
    let single = OutputCodeClassifier::new(mean_factory()).fit(&x3, &y1);
    assert!(single.is_err());
}

// ---------------------------------------------------------------------------
// REQ-OCC-NCODES (#1830) — DETERMINISTIC FIXABLE DIVERGENCE. FAILING PIN.
//
// sklearn `n_estimators = int(n_classes * self.code_size)`  (FLOOR via int(),
// NO max(2)) — sklearn/multiclass.py:1189.
// ferrolearn `n_codes = (code_size*k).ceil().max(2)` — chain.rs:371.
//
// LIVE ORACLE (R-CHAR-3): for k=3 classes, code_size=1.5:
//   cd /tmp && python3 -c "
//   import numpy as np
//   from sklearn.multiclass import OutputCodeClassifier
//   from sklearn.ensemble import RandomForestClassifier
//   X = np.repeat(np.arange(3),3).reshape(-1,1).astype(float)
//   y = np.repeat(np.arange(3),3)
//   occ = OutputCodeClassifier(RandomForestClassifier(random_state=0),
//                              code_size=1.5, random_state=0).fit(X, y)
//   print(len(occ.estimators_))"
//   # -> 4        (int(3*1.5)=int(4.5)=4)
// ferrolearn: ceil(4.5).max(2) == 5  =>  this assertion FAILS today.
// ---------------------------------------------------------------------------
#[test]
#[ignore = "divergence: OCC n_codes ceil().max(2) vs sklearn int() floor; tracking #1830"]
fn divergence_occ_n_codes_floor() {
    let x = Array2::<f64>::zeros((9, 2));
    let y = Array1::from(vec![0usize, 0, 0, 1, 1, 1, 2, 2, 2]); // k = 3
    let fitted = OutputCodeClassifier::new(mean_factory())
        .code_size(1.5)
        .random_state(0)
        .fit(&x, &y)
        .unwrap();
    // Live oracle: len(estimators_) == int(3 * 1.5) == 4.
    const SK_N_ESTIMATORS: usize = 4; // sklearn/multiclass.py:1189, int(3*1.5)
    assert_eq!(
        fitted.n_estimators(),
        SK_N_ESTIMATORS,
        "OCC n_codes diverges: ferrolearn ceil(4.5).max(2)=5 vs sklearn int(4.5)=4"
    );
}

// ---------------------------------------------------------------------------
// REQ-OCC-NCODES coupled validation facet (#1830) — code_size <= 0. FAILING PIN.
//
// sklearn constraint `Interval(Real, 0.0, None, closed="neither")`
// (sklearn/multiclass.py:1130) raises InvalidParameterError at fit for
// code_size == 0.0 (and negatives).
// LIVE ORACLE (R-CHAR-3):
//   OutputCodeClassifier(RandomForestClassifier(), code_size=0.0).fit(X,y)
//   # -> raises InvalidParameterError
//   OutputCodeClassifier(RandomForestClassifier(), code_size=-1.0).fit(X,y)
//   # -> raises InvalidParameterError
// ferrolearn `.max(2)` MASKS this: code_size=0.0 yields n_codes=2 and fits
// successfully (no error) => this assertion FAILS today.
// ---------------------------------------------------------------------------
#[test]
#[ignore = "divergence: OCC code_size<=0 masked by .max(2); sklearn raises InvalidParameterError; tracking #1830"]
fn divergence_occ_code_size_zero_should_error() {
    let x = Array2::<f64>::zeros((9, 2));
    let y = Array1::from(vec![0usize, 0, 0, 1, 1, 1, 2, 2, 2]);
    // sklearn raises InvalidParameterError for code_size == 0.0.
    let result = OutputCodeClassifier::new(mean_factory())
        .code_size(0.0)
        .random_state(0)
        .fit(&x, &y);
    assert!(
        result.is_err(),
        "code_size=0.0 should error (sklearn InvalidParameterError, \
         multiclass.py:1130); ferrolearn's .max(2) masks it into n_codes=2"
    );
}
