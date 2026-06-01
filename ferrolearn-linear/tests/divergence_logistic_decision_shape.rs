//! Divergence: ferrolearn's `FittedLogisticRegression::decision_function`
//! diverges from sklearn `LogisticRegression.decision_function` (inherited from
//! `LinearClassifierMixin`) in OUTPUT SHAPE for the binary case.
//!
//! sklearn `sklearn/linear_model/_base.py:365`:
//!   `return xp.reshape(scores, (-1,)) if scores.shape[1] == 1 else scores`
//! i.e. the binary `(n_samples, 1)` score matrix is RAVELED to a 1-D
//! `(n_samples,)` array (docstring `_base.py:355`:
//!   `scores : ndarray of shape (n_samples,) or (n_samples, n_classes)`).
//!
//! Live oracle (sklearn 1.5.2):
//!   X=[[1,2],[2,3],[3,4],[5,6],[6,7],[7,8]], y=[0,0,0,1,1,1], C=1.0:
//!   `m.decision_function(X).shape == (6,)`, `.ndim == 1`.
//!
//! ferrolearn `decision_function in logistic_regression.rs` (binary branch)
//! returns `Array2<F>` of shape `(n_samples, 1)` — ndim 2, an extra trailing
//! axis sklearn does not have. This violates R-DEV-3 (returned-shape contract):
//! a user comparing `ferrolearn.decision_function(X)` to
//! `sklearn.decision_function(X)` array-by-array sees a rank mismatch.
//!
//! Tracking: #454
use ferrolearn_core::traits::Fit;
use ferrolearn_linear::LogisticRegression;
use ndarray::{array, Array2};

#[test]
fn divergence_binary_decision_function_shape() {
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![1., 2., 2., 3., 3., 4., 5., 6., 6., 7., 7., 8.],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 1, 1, 1];

    let fitted = LogisticRegression::<f64>::new()
        .with_c(1.0)
        .with_max_iter(1000)
        .fit(&x, &y)
        .unwrap();

    let df = fitted.decision_function(&x).unwrap();

    // sklearn oracle: binary decision_function is 1-D, shape (6,).
    // `_base.py:365` ravels the (n,1) score to (-1,). ferrolearn keeps
    // ndim==2 / ncols==1, so this assertion FAILS against current behavior.
    const SK_NDIM: usize = 1; // sklearn `decision_function(X).ndim == 1` (binary)
    assert_eq!(
        df.ndim(),
        SK_NDIM,
        "binary decision_function must be 1-D per sklearn _base.py:365 ravel; \
         got shape {:?}",
        df.shape()
    );
}
