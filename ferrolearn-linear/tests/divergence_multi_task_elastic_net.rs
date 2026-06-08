//! Divergence pins for `MultiTaskElasticNet` (REQ-14, #21) vs scikit-learn
//! 1.5.2 `sklearn.linear_model.MultiTaskElasticNet`.
//!
//! `MultiTaskElasticNet` is SHIPPED but had no dedicated divergence-test file
//! and was never adversarially audited. Its sibling `MultiTaskLasso` was just
//! audited and fixed for TWO gaps (non-finite-input rejection + missing
//! `dual_gap_`). This file pins the SAME two gaps for `MultiTaskElasticNet`,
//! which still has neither.
//!
//! Audit scope confirmed PARITY (no pins needed) for:
//!   * coef_ / intercept_ / n_iter_ across the l1_ratio {0.1,0.3,0.5,0.7,0.9,1.0}
//!     x alpha {0.05,0.2,0.5,1.0} grid — every value matches the live oracle
//!     bit-for-bit (probed via the in-tree exploration harness, then removed).
//!   * l1_ratio bounds: sklearn `Interval(Real, 0, 1, closed="both")`
//!     (`ElasticNet._parameter_constraints`, `_coordinate_descent.py:2320`);
//!     ferrolearn rejects `l1_ratio < 0 || > 1` (multi_task_elastic_net.rs:231)
//!     — matches, already covered by `multi_task_elastic_net_validation_errors`.
//!   * l1_ratio=1.0 == MultiTaskLasso — covered by the in-module test.

use ferrolearn_core::traits::Fit;
use ferrolearn_linear::MultiTaskElasticNet;
use ndarray::{Array2, array};

fn fixture() -> (Array2<f64>, Array2<f64>) {
    let x: Array2<f64> = array![[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 5.0]];
    let y: Array2<f64> = array![[3.0, 1.0], [2.5, 2.0], [7.1, 3.5], [6.0, 4.2], [11.2, 6.0]];
    (x, y)
}

/// Divergence (#2, KNOWN sibling gap): `MultiTaskElasticNet::fit` performs no
/// finiteness validation, while sklearn rejects NaN/Inf.
///
/// sklearn `MultiTaskElasticNet.fit` calls `self._validate_data(X, y,
/// validate_separately=(check_X_params, check_y_params))`
/// (`sklearn/linear_model/_coordinate_descent.py:2602`). Both param dicts
/// (`:2592`, `:2598`) inherit the default `force_all_finite=True`, so any NaN
/// or +/-Inf in X or Y raises `ValueError` BEFORE the solver runs.
///
/// Live sklearn 1.5.2 oracle (R-CHAR-3), never copied from ferrolearn:
///   python3 -c "import numpy as np;
///     from sklearn.linear_model import MultiTaskElasticNet;
///     X=np.array([[1,2],[2,1],[3,4],[4,3],[5,5]],float);
///     Y=np.array([[3,1],[2.5,2],[7.1,3.5],[6,4.2],[11.2,6]]);
///     Xn=X.copy(); Xn[0,0]=np.nan;
///     MultiTaskElasticNet(alpha=0.3,l1_ratio=0.5).fit(Xn,Y)"
///   -> ValueError: Input X contains NaN.
///   Xi[1,0]=np.inf -> ValueError: Input X contains infinity ...
///   Yi[0,0]=np.inf -> ValueError: Input y contains infinity ...
///
/// ferrolearn `multi_task_elastic_net.rs::fit` has NO finiteness check (unlike
/// the just-fixed `MultiTaskLasso` sibling at `multi_task_lasso.rs:239-250`):
/// it centers, runs block CD, and returns `Ok(FittedMultiTaskElasticNet)` whose
/// `coef_` is NaN-poisoned. R-DEV-2 error-contract divergence.
///
/// Tracking: #2240
#[test]
#[ignore = "divergence: MultiTaskElasticNet::fit accepts non-finite X/Y (sklearn raises ValueError); tracking #2240"]
fn mten_rejects_non_finite_input_like_sklearn() {
    // 1. NaN in X: sklearn raises ValueError; ferrolearn must Err, not Ok.
    let (mut x, y) = fixture();
    x[[0, 0]] = f64::NAN;
    let res = MultiTaskElasticNet::<f64>::new()
        .with_alpha(0.3)
        .with_l1_ratio(0.5)
        .fit(&x, &y);
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
    let res = MultiTaskElasticNet::<f64>::new()
        .with_alpha(0.3)
        .with_l1_ratio(0.5)
        .fit(&x, &y);
    assert!(
        res.is_err(),
        "sklearn raises ValueError for Inf in X; ferrolearn must return Err"
    );

    // 3. -Inf in X: sklearn raises ValueError; ferrolearn must Err.
    let (mut x, y) = fixture();
    x[[2, 1]] = f64::NEG_INFINITY;
    let res = MultiTaskElasticNet::<f64>::new()
        .with_alpha(0.3)
        .with_l1_ratio(0.5)
        .fit(&x, &y);
    assert!(
        res.is_err(),
        "sklearn raises ValueError for -Inf in X; ferrolearn must return Err"
    );

    // 4. NaN in Y: sklearn raises ValueError; ferrolearn must Err.
    let (x, mut y) = fixture();
    y[[2, 1]] = f64::NAN;
    let res = MultiTaskElasticNet::<f64>::new()
        .with_alpha(0.3)
        .with_l1_ratio(0.5)
        .fit(&x, &y);
    assert!(
        res.is_err(),
        "sklearn raises ValueError for NaN in Y; ferrolearn must return Err"
    );

    // 5. Inf in Y: sklearn raises ValueError; ferrolearn must Err.
    let (x, mut y) = fixture();
    y[[0, 0]] = f64::INFINITY;
    let res = MultiTaskElasticNet::<f64>::new()
        .with_alpha(0.3)
        .with_l1_ratio(0.5)
        .fit(&x, &y);
    assert!(
        res.is_err(),
        "sklearn raises ValueError for Inf in Y; ferrolearn must return Err"
    );
}
