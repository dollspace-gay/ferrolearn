//! Regression guard for the Rust LIBRARY contract of
//! `FittedLogisticRegression::decision_function`, plus the binding-ABI
//! tracking note for #454.
//!
//! ## Two distinct contracts
//!
//! **Rust library (this crate):** `decision_function -> Array2<F>` returns a
//! uniform `(n_samples, 1)` for the binary case (and `(n_samples, n_classes)`
//! for multiclass). This 2-D shape is a legitimate Rust idiom — callers pattern
//! on `Array2` regardless of class count, matching the tree-crate convention.
//! The library contract is correct and is NOT changed here.
//!
//! **ferrolearn-python binding ABI (tracked by #454):** sklearn
//! `LinearClassifierMixin.decision_function` (`sklearn/linear_model/_base.py:365`)
//! ravels the binary `(n_samples, 1)` score matrix to a 1-D `(n_samples,)`
//! array (docstring `_base.py:355`:
//!   `scores : ndarray of shape (n_samples,) or (n_samples, n_classes)`).
//! Per goal.md, correctness lives in the library crate; only marshalling/ABI
//! concerns — including SHAPE at the Python↔Rust boundary — are fixed in
//! `ferrolearn-python`. So the sklearn `(n,)` binary shape is a BINDING ABI
//! contract: `ferrolearn.LogisticRegression.decision_function` must ravel
//! `(n,1)->(n,)` to mirror `import sklearn`. That is tracked by #454 and will
//! be pinned as a pytest divergence when the loop reaches `ferrolearn-python`.
//! (Same pattern as the RidgeClassifier `coef_` orientation binding fix.)
//!
//! This guard therefore asserts the LIBRARY contract: the binary
//! `decision_function` returns `Array2` shape `(n, 1)` AND its VALUES match the
//! live sklearn 1.5.2 oracle's `decision_function` (sklearn `(n,)` raveled —
//! the values are identical, only the axis differs).
//!
//! Tracking: #454 (ferrolearn-python binding ABI item; kept open).
use ferrolearn_core::traits::Fit;
use ferrolearn_linear::LogisticRegression;
use ndarray::{Array2, array};

#[test]
fn logistic_decision_function_values_match_sklearn_rust_array2_shape() {
    let x = Array2::from_shape_vec((6, 2), vec![1., 2., 2., 3., 3., 4., 5., 6., 6., 7., 7., 8.])
        .unwrap();
    let y = array![0usize, 0, 0, 1, 1, 1];

    let fitted = LogisticRegression::<f64>::new()
        .with_c(1.0)
        .with_max_iter(1000)
        .fit(&x, &y)
        .unwrap();

    let df = fitted.decision_function(&x).unwrap();

    // LIBRARY contract: the Rust binary decision_function returns a uniform
    // 2-D Array2 of shape (n, 1). The sklearn (n,) binary shape is enforced at
    // the ferrolearn-python binding boundary (#454), NOT here.
    assert_eq!(df.ndim(), 2, "Rust library decision_function is Array2");
    assert_eq!(
        df.shape(),
        &[6, 1],
        "Rust binary decision_function shape is (n, 1); the sklearn (n,) ravel \
         is a ferrolearn-python binding ABI concern tracked by #454"
    );

    // VALUES contract (R-DEV-3 sign convention + magnitude): identical to the
    // live sklearn 1.5.2 oracle's decision_function, which for binary is the
    // raveled (n,) of the same scores. Expected values from the live oracle
    // (R-CHAR-3 — NOT copied from ferrolearn):
    //   python3 -c "from sklearn.linear_model import LogisticRegression;
    //     import numpy as np;
    //     X=np.array([[1.,2.],[2.,3.],[3.,4.],[5.,6.],[6.,7.],[7.,8.]]);
    //     y=np.array([0,0,0,1,1,1]);
    //     m=LogisticRegression(C=1.0,max_iter=1000).fit(X,y);
    //     print(m.decision_function(X).tolist())"
    // -> [-4.227389544735363, -2.8182084514613215, -1.409027358187279,
    //     1.4093348283608043, 2.8185159216348463, 4.227697014908889]
    let sklearn_oracle = [
        -4.227389544735363,
        -2.8182084514613215,
        -1.409027358187279,
        1.4093348283608043,
        2.8185159216348463,
        4.227697014908889,
    ];

    // Compare element-wise. Tolerance is convergence-scale (~1e-3): both
    // implementations solve the same regularized logistic objective with
    // independent optimizers, so the raw scores agree to optimizer tolerance.
    for (i, &expected) in sklearn_oracle.iter().enumerate() {
        let got = df[[i, 0]];
        assert!(
            (got - expected).abs() < 1e-3,
            "row {i}: ferrolearn decision_function {got} differs from sklearn \
             oracle {expected} beyond convergence tolerance"
        );
    }
}
