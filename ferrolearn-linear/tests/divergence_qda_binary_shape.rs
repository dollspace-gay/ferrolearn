//! #581 — binary `decision_function` shape/sign for
//! `QuadraticDiscriminantAnalysis`, mirroring the logistic #454 precedent
//! (`divergence_logistic_decision_shape.rs`).
//!
//! ## Two distinct contracts
//!
//! **Rust library (this crate):** `FittedQDA::decision_function -> Array2<F>`
//! ALWAYS returns the raw `(n_samples, n_classes)` per-class log-posterior — for
//! binary that is `(n, 2)` (the two raw class scores, NOT their difference).
//! This uniform 2-D shape is the same Rust idiom the logistic library uses
//! (`divergence_logistic_decision_shape.rs`): callers pattern on `Array2`
//! regardless of class count.
//!
//! **scikit-learn (the oracle):** `QuadraticDiscriminantAnalysis.decision_function`
//! collapses the BINARY case to shape `(n,)` = `dec[:,1] - dec[:,0]`, the
//! log-likelihood ratio of the positive class
//! (`sklearn/discriminant_analysis.py:1000-1002`:
//!   `if len(self.classes_) == 2: return dec_func[:, 1] - dec_func[:, 0]`).
//!
//! Per goal.md, correctness lives in the library crate; SHAPE at the Python↔Rust
//! boundary is a marshalling/ABI concern fixed in `ferrolearn-python`. So the
//! sklearn `(n,)` binary shape is a BINDING ABI contract:
//! `ferrolearn.QuadraticDiscriminantAnalysis.decision_function` must transform
//! the library's binary `(n,2)` matrix into sklearn's `(n,)` = `col1 - col0`.
//! That ravel-and-difference is tracked by #581 and will be pinned as a pytest
//! divergence when the loop reaches `ferrolearn-python` (the binding currently
//! exposes only `new`/`fit`/`predict`, so `decision_function` is not yet on the
//! ABI at all).
//!
//! NOTE vs logistic #454: the logistic library returns binary `(n, 1)` (a single
//! raveled score), so its binding ravel is a pure `(n,1)->(n,)` reshape. QDA's
//! library returns `(n, 2)` (two raw scores), so its binding transform is
//! `dec[:,1] - dec[:,0]` — a value combination, not just a reshape. This test
//! pins BOTH facts: the library shape is `(n, 2)`, and `col1 - col0` reproduces
//! the live sklearn `(n,)` binary `decision_function` exactly.
//!
//! Tracking: #581 (ferrolearn-python binding ABI item; kept open).

use ferrolearn_core::traits::Fit;
use ferrolearn_linear::QDA;
use ndarray::{Array1, Array2};

/// Library contract: binary `decision_function` is `Array2` shape `(n, 2)`, and
/// the binding's required `dec[:,1] - dec[:,0]` transform reproduces the live
/// sklearn 1.5.2 binary `decision_function` `(n,)` values exactly.
///
/// Oracle (live sklearn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; from sklearn.discriminant_analysis import \
///   QuadraticDiscriminantAnalysis as Q; \
///   X=np.array([[1.,1.],[1.,3.],[2.,1.5],[2.,2.5],[8.,8.],[8.,9.5],[9.,8.5],[9.,9.5]]); \
///   y=np.array([0,0,0,0,1,1,1,1]); m=Q().fit(X,y); \
///   print(m.decision_function(X).shape); print(m.decision_function(X).tolist())"
/// # shape (8,)
/// # [-116.41537777272299, -97.95383931118454, -93.15383931118453,
/// #  -83.46153161887683, 84.29616068881546, 96.44616068881548,
/// #  109.33462222727701, 117.73462222727701]
/// ```
#[test]
fn qda_binary_decision_function_shape() {
    // Live sklearn 1.5.2 binary `decision_function(X)` (shape (8,), = col1-col0).
    const SK_BINARY_DEC: [f64; 8] = [
        -116.41537777272299,
        -97.95383931118454,
        -93.15383931118453,
        -83.46153161887683,
        84.29616068881546,
        96.44616068881548,
        109.33462222727701,
        117.73462222727701,
    ];

    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            1.0, 1.0, 1.0, 3.0, 2.0, 1.5, 2.0, 2.5, // class 0
            8.0, 8.0, 8.0, 9.5, 9.0, 8.5, 9.0, 9.5, // class 1
        ],
    )
    .unwrap();
    let y = Array1::from(vec![0, 0, 0, 0, 1, 1, 1, 1]);

    let fitted = QDA::<f64>::new().fit(&x, &y).unwrap();
    let dec = fitted.decision_function(&x).unwrap();

    // LIBRARY contract: ferrolearn's binary decision_function is the raw (n, 2)
    // matrix. The sklearn (n,) collapse is a ferrolearn-python binding ABI
    // concern tracked by #581, NOT enforced in the library here.
    assert_eq!(
        dec.dim(),
        (8, 2),
        "Rust library binary decision_function shape is (n, 2) (raw per-class \
         scores); the sklearn (n,) = dec[:,1]-dec[:,0] collapse is a \
         ferrolearn-python binding ABI concern tracked by #581"
    );

    // BINDING transform (the ravel #581 must implement): col1 - col0 reproduces
    // the live sklearn binary decision_function (n,) values exactly. Expected
    // values from the live oracle (R-CHAR-3 — NOT copied from ferrolearn).
    for i in 0..8 {
        let diff = dec[[i, 1]] - dec[[i, 0]];
        assert!(
            (diff - SK_BINARY_DEC[i]).abs() < 1e-6,
            "binary decision_function[{i}]: sklearn (col1-col0) {}, ferrolearn \
             dec[{i},1]-dec[{i},0] {}",
            SK_BINARY_DEC[i],
            diff
        );
    }
}
