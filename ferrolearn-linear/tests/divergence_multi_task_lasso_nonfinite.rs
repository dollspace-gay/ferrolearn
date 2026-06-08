//! Divergence pin: `MultiTaskLasso::fit` non-finite-input rejection vs
//! scikit-learn 1.5.2.
//!
//! sklearn's `MultiTaskElasticNet.fit` (the base of `MultiTaskLasso`) calls
//! `self._validate_data(X, y, validate_separately=(check_X_params,
//! check_y_params))` at `sklearn/linear_model/_coordinate_descent.py:2602`.
//! Both `check_X_params` (`:2595`) and `check_y_params` (`:2601`) inherit
//! sklearn's default `force_all_finite=True`, so any NaN or Inf in `X` or `y`
//! raises `ValueError("Input X contains infinity or a value too large ...")`
//! BEFORE the solver runs.
//!
//! ferrolearn's `multi_task_lasso.rs::fit` performs no finiteness validation:
//! it centers, runs block CD, and returns `Ok(FittedMultiTaskLasso)` whose
//! `coef_` is NaN-poisoned. That is an R-DEV-2 error-contract divergence —
//! sklearn errors, ferrolearn silently returns garbage.
//!
//! Live sklearn 1.5.2 oracle (R-CHAR-3), never copied from ferrolearn:
//!   python3 -c "import warnings; warnings.filterwarnings('ignore');
//!     from sklearn.linear_model import MultiTaskLasso; import numpy as np;
//!     X=np.array([[1,2],[2,1],[3,4],[4,3],[5,5]],float);
//!     Y=np.array([[3,1],[2.5,2],[7.1,3.5],[6,4.2],[11.2,6]]);
//!     Xn=X.copy(); Xn[0,0]=np.nan;
//!     MultiTaskLasso(alpha=0.3).fit(Xn,Y)"
//!   -> ValueError: Input X contains NaN.
//! With Xn[0,0]=np.inf -> ValueError: Input X contains infinity ...
//! With Y[0,0]=np.inf  -> ValueError: Input y contains infinity ...
//!
//! ferrolearn current behavior (probe): the NaN-in-X fit returns
//! `Ok(..)` with `coef_ = [[NaN, NaN], [NaN, NaN]]`.
//!
//! Tracking: #2

use ferrolearn_core::traits::Fit;
use ferrolearn_linear::MultiTaskLasso;
use ndarray::{Array2, array};

fn fixture() -> (Array2<f64>, Array2<f64>) {
    let x: Array2<f64> = array![[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 5.0]];
    let y: Array2<f64> = array![[3.0, 1.0], [2.5, 2.0], [7.1, 3.5], [6.0, 4.2], [11.2, 6.0]];
    (x, y)
}

#[test]
#[ignore = "divergence: MultiTaskLasso::fit accepts non-finite X/Y while sklearn _coordinate_descent.py:2602 raises ValueError; tracking #2"]
fn mtl_rejects_non_finite_input_like_sklearn() {
    // 1. NaN in X: sklearn raises ValueError; ferrolearn must Err, not Ok.
    let (mut x, y) = fixture();
    x[[0, 0]] = f64::NAN;
    let res = MultiTaskLasso::<f64>::new().with_alpha(0.3).fit(&x, &y);
    assert!(
        res.is_err(),
        "sklearn raises ValueError for NaN in X \
         (_coordinate_descent.py:2602); ferrolearn must return Err, got Ok \
         with coef_={:?}",
        res.ok().map(|f| f.coefficients().clone())
    );

    // 2. Inf in X: sklearn raises ValueError; ferrolearn must Err.
    let (mut x, y) = fixture();
    x[[1, 0]] = f64::INFINITY;
    let res = MultiTaskLasso::<f64>::new().with_alpha(0.3).fit(&x, &y);
    assert!(
        res.is_err(),
        "sklearn raises ValueError for Inf in X; ferrolearn must return Err"
    );

    // 3. NaN in Y: sklearn raises ValueError; ferrolearn must Err.
    let (x, mut y) = fixture();
    y[[2, 1]] = f64::NAN;
    let res = MultiTaskLasso::<f64>::new().with_alpha(0.3).fit(&x, &y);
    assert!(
        res.is_err(),
        "sklearn raises ValueError for NaN in Y; ferrolearn must return Err"
    );
}
