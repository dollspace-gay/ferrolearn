//! Divergence / conformance guards for `ferrolearn-model-sel::multioutput`
//! against scikit-learn 1.5.2 `sklearn.multioutput`.
//!
//! All expected values are produced by the LIVE sklearn 1.5.2 oracle (run from
//! `/tmp`), never copied from the ferrolearn side (R-CHAR-3). The oracle
//! commands are reproduced in each guard's doc-comment so the constant is
//! traceable.
//!
//! Mirrors:
//! - `_MultiOutputEstimator.fit`   `sklearn/multioutput.py:216-290`
//! - `_MultiOutputEstimator.predict` `sklearn/multioutput.py:292-314`
//! - `_fit_estimator`              `sklearn/multioutput.py:62-68`
//! - `MultiOutputRegressor`        `sklearn/multioutput.py:342`
//! - `MultiOutputClassifier`       `sklearn/multioutput.py:445`
//!
//! GREEN guards: would FAIL if the SHIPPED claim broke.
//! There are NO `#[ignore]`d pins below: every audited corner of the
//! IMPLEMENTED surface matched the oracle (see report). The NOT-STARTED REQs
//! (predict_proba/classes_/score/sample_weight/partial_fit/n_jobs) are
//! whole-feature blockers, not single-file divergences, so they are not pinned.

use ferrolearn_core::pipeline::{FittedPipelineEstimator, Pipeline, PipelineEstimator};
use ferrolearn_core::{FerroError, Predict};
use ferrolearn_model_sel::multioutput::{MultiOutputClassifier, MultiOutputRegressor};
use ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// Base estimator fixtures
// ---------------------------------------------------------------------------

/// Predicts the training-target mean for every row.
///
/// sklearn analog: `DummyRegressor(strategy='mean')` (regression) /
/// `DummyClassifier(strategy='most_frequent')`-style constant per column.
struct MeanEstimator;
struct FittedMeanEst {
    mean: f64,
}

impl PipelineEstimator<f64> for MeanEstimator {
    fn fit_pipeline(
        &self,
        _x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<Box<dyn FittedPipelineEstimator<f64>>, FerroError> {
        Ok(Box::new(FittedMeanEst {
            mean: y.mean().unwrap_or(0.0),
        }))
    }
}
impl FittedPipelineEstimator<f64> for FittedMeanEst {
    fn predict_pipeline(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
        Ok(Array1::from_elem(x.nrows(), self.mean))
    }
}

/// Exact ordinary-least-squares base for a single output column (2 features +
/// intercept, fit by normal equations). Used to reproduce sklearn's
/// `LinearRegression` per-column exact recovery without noise so the predicted
/// `(n_samples, n_outputs)` matrix can be matched element-for-element. Requires
/// an over-determined fixture (n_samples > 3) so the 3x3 normal system is
/// full-rank.
struct OlsEstimator;
struct FittedOls {
    coef: [f64; 2],
    intercept: f64,
}

impl PipelineEstimator<f64> for OlsEstimator {
    fn fit_pipeline(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<Box<dyn FittedPipelineEstimator<f64>>, FerroError> {
        // Build design matrix [1, x0, x1] and solve 3x3 normal equations.
        let n = x.nrows();
        let mut xt_x = [[0.0f64; 3]; 3];
        let mut xt_y = [0.0f64; 3];
        for i in 0..n {
            let row = [1.0, x[[i, 0]], x[[i, 1]]];
            for a in 0..3 {
                xt_y[a] += row[a] * y[i];
                for b in 0..3 {
                    xt_x[a][b] += row[a] * row[b];
                }
            }
        }
        let beta = solve3(xt_x, xt_y);
        Ok(Box::new(FittedOls {
            intercept: beta[0],
            coef: [beta[1], beta[2]],
        }))
    }
}
impl FittedPipelineEstimator<f64> for FittedOls {
    fn predict_pipeline(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
        let preds: Vec<f64> = (0..x.nrows())
            .map(|i| self.intercept + self.coef[0] * x[[i, 0]] + self.coef[1] * x[[i, 1]])
            .collect();
        Ok(Array1::from_vec(preds))
    }
}

/// Gaussian-elimination solve of a 3x3 system (well-conditioned fixtures only).
#[allow(
    clippy::needless_range_loop,
    reason = "explicit row/col indices read as textbook Gaussian elimination"
)]
fn solve3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> [f64; 3] {
    for col in 0..3 {
        // Partial pivot.
        let mut piv = col;
        for r in (col + 1)..3 {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        a.swap(col, piv);
        b.swap(col, piv);
        let d = a[col][col];
        for r in 0..3 {
            if r == col {
                continue;
            }
            let f = a[r][col] / d;
            for c in 0..3 {
                a[r][c] -= f * a[col][c];
            }
            b[r] -= f * b[col];
        }
    }
    [b[0] / a[0][0], b[1] / a[1][1], b[2] / a[2][2]]
}

fn mean_factory() -> Box<dyn Fn() -> Pipeline<f64> + Send + Sync> {
    Box::new(|| Pipeline::new().estimator_step("est", Box::new(MeanEstimator)))
}
fn ols_factory() -> Box<dyn Fn() -> Pipeline<f64> + Send + Sync> {
    Box::new(|| Pipeline::new().estimator_step("est", Box::new(OlsEstimator)))
}

// ===========================================================================
// REQ-FIT-PER-COLUMN — one estimator per output column (n_estimators == K)
// ===========================================================================

/// Guard: REQ-FIT-PER-COLUMN. K-column `y` ⇒ `n_estimators() == K == n_targets()`.
///
/// Mirrors `self.estimators_ = Parallel(...)(_fit_estimator(self.estimator, X,
/// y[:, i]) for i in range(y.shape[1]))` `sklearn/multioutput.py:278-283`,
/// fresh `clone` per column `sklearn/multioutput.py:62-68`.
///
/// LIVE ORACLE (run from /tmp):
///   `len(MultiOutputRegressor(LinearRegression()).fit(X, Y).estimators_)` == y.shape[1].
#[test]
fn guard_fit_per_column_count() {
    // K = 3 columns.
    let x = Array2::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap();
    let y = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]).unwrap();

    let mor = MultiOutputRegressor::new(mean_factory());
    let fitted = mor.fit(&x, &y).unwrap();
    assert_eq!(fitted.n_targets(), 3, "one estimator per output column");
    assert_eq!(fitted.n_estimators(), 3, "n_estimators == n_targets");

    let moc = MultiOutputClassifier::new(mean_factory());
    let yc = Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 0.0, 1.0]).unwrap();
    let fc = moc.fit(&x, &yc).unwrap();
    assert_eq!(fc.n_estimators(), 2);
}

// ===========================================================================
// REQ-PREDICT-STACK — column-stack, original order, (n_samples, n_targets)
// ===========================================================================

/// Guard: REQ-PREDICT-STACK end-to-end vs the oracle with an EXACT OLS base.
///
/// Mirrors `np.asarray([e.predict(X) for e in estimators_]).T`
/// `sklearn/multioutput.py:310-314`.
///
/// LIVE ORACLE (run from /tmp):
/// ```text
/// X = np.array([[1.,0.],[0.,1.],[1.,1.],[2.,1.],[0.,2.],[2.,2.]])
/// Y = np.column_stack([2*X[:,0]+3*X[:,1]+1.0, -1*X[:,0]+4*X[:,1]])
/// P = MultiOutputRegressor(LinearRegression()).fit(X, Y).predict(X)
/// # P.shape == (6,2); np.allclose(P, Y) == True
/// ```
/// The defining functions are y0 = 2*x0 + 3*x1 + 1 and y1 = -1*x0 + 4*x1,
/// so the predicted matrix must equal `Y` exactly (oracle: allclose True).
#[test]
fn guard_predict_stack_exact_ols() {
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.0, 2.0, 2.0, 2.0],
    )
    .unwrap();
    // Expected targets defined by the SAME two linear functions the oracle uses.
    let mut y = Array2::<f64>::zeros((6, 2));
    for i in 0..6 {
        let (x0, x1) = (x[[i, 0]], x[[i, 1]]);
        y[[i, 0]] = 2.0 * x0 + 3.0 * x1 + 1.0;
        y[[i, 1]] = -x0 + 4.0 * x1;
    }

    let mor = MultiOutputRegressor::new(ols_factory());
    let fitted = mor.fit(&x, &y).unwrap();
    let p = fitted.predict(&x).unwrap();

    assert_eq!(p.shape(), &[6, 2], "predict shape (n_samples, n_targets)");
    for i in 0..6 {
        assert!(
            (p[[i, 0]] - y[[i, 0]]).abs() < 1e-9,
            "col0 exact recovery at row {i}: {} vs {}",
            p[[i, 0]],
            y[[i, 0]]
        );
        assert!(
            (p[[i, 1]] - y[[i, 1]]).abs() < 1e-9,
            "col1 exact recovery at row {i}: {} vs {}",
            p[[i, 1]],
            y[[i, 1]]
        );
    }
}

/// Guard: REQ-PREDICT-STACK COLUMN-ORDER preservation (adversarial transpose
/// stress). Uses distinct per-column constants so a swapped/transposed result
/// is detectable.
///
/// LIVE ORACLE (run from /tmp):
/// ```text
/// X = np.array([[1.,2.],[3.,4.],[5.,6.],[7.,8.]])
/// Y = np.array([[2.,4.],[2.,4.],[2.,4.],[2.,4.]])
/// P = MultiOutputRegressor(DummyRegressor('mean')).fit(X, Y).predict(X)
/// # P.shape == (4,2); P[0].tolist() == [2.0, 4.0]   (NOT [4.0, 2.0])
/// ```
/// col0 mean = 2.0, col1 mean = 4.0. Asserts result[:,0]==2.0, result[:,1]==4.0.
#[test]
fn guard_predict_stack_column_order() {
    let x = Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
    let y = Array2::from_shape_vec((4, 2), vec![2.0, 4.0, 2.0, 4.0, 2.0, 4.0, 2.0, 4.0]).unwrap();

    let mor = MultiOutputRegressor::new(mean_factory());
    let p = mor.fit(&x, &y).unwrap().predict(&x).unwrap();

    assert_eq!(p.shape(), &[4, 2]);
    for i in 0..4 {
        // Oracle: P[i] == [2.0, 4.0]. A transpose/swap bug would give [4.0, 2.0].
        assert!((p[[i, 0]] - 2.0).abs() < 1e-10, "col0 must be 2.0 (oracle)");
        assert!((p[[i, 1]] - 4.0).abs() < 1e-10, "col1 must be 4.0 (oracle)");
    }
}

/// Guard: REQ-PREDICT-STACK column-order under an n_samples == n_outputs == 2
/// SQUARE result (where an accidental missing/extra transpose is invisible to a
/// shape check). Distinct per-column means make a transpose detectable.
///
/// LIVE ORACLE (run from /tmp):
/// ```text
/// X = np.array([[1.,9.],[2.,8.]])
/// Y = np.array([[2.,4.],[2.,4.]])   # col0 mean 2, col1 mean 4
/// P = MultiOutputRegressor(DummyRegressor('mean')).fit(X, Y).predict(X)
/// # P.tolist() == [[2.0, 4.0], [2.0, 4.0]]   (a transpose bug -> [[2,2],[4,4]])
/// ```
#[test]
fn guard_predict_stack_square_no_transpose_bug() {
    let x = Array2::from_shape_vec((2, 2), vec![1.0, 9.0, 2.0, 8.0]).unwrap();
    let y = Array2::from_shape_vec((2, 2), vec![2.0, 4.0, 2.0, 4.0]).unwrap();

    let mor = MultiOutputRegressor::new(mean_factory());
    let p = mor.fit(&x, &y).unwrap().predict(&x).unwrap();
    // Oracle P == [[2,4],[2,4]]. A transpose bug would give [[2,2],[4,4]].
    assert!((p[[0, 0]] - 2.0).abs() < 1e-10);
    assert!((p[[0, 1]] - 4.0).abs() < 1e-10);
    assert!((p[[1, 0]] - 2.0).abs() < 1e-10);
    assert!((p[[1, 1]] - 4.0).abs() < 1e-10);
}

/// Guard: REQ-PREDICT-STACK column INDEPENDENCE — swapping the output columns
/// permutes the predicted columns identically (no cross-column coupling).
///
/// LIVE ORACLE (run from /tmp):
/// ```text
/// P  = MultiOutputRegressor(LinearRegression()).fit(X, Y).predict(X)
/// P2 = MultiOutputRegressor(LinearRegression()).fit(X, Y[:, ::-1]).predict(X)
/// # np.allclose(P2, P[:, ::-1]) == True
/// ```
#[test]
fn guard_predict_stack_column_independence() {
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.0, 2.0, 2.0, 2.0],
    )
    .unwrap();
    let mut y = Array2::<f64>::zeros((6, 2));
    let mut y_swapped = Array2::<f64>::zeros((6, 2));
    for i in 0..6 {
        let (x0, x1) = (x[[i, 0]], x[[i, 1]]);
        let c0 = 2.0 * x0 + 3.0 * x1 + 1.0;
        let c1 = -x0 + 4.0 * x1;
        y[[i, 0]] = c0;
        y[[i, 1]] = c1;
        y_swapped[[i, 0]] = c1;
        y_swapped[[i, 1]] = c0;
    }

    let p = MultiOutputRegressor::new(ols_factory())
        .fit(&x, &y)
        .unwrap()
        .predict(&x)
        .unwrap();
    let p2 = MultiOutputRegressor::new(ols_factory())
        .fit(&x, &y_swapped)
        .unwrap()
        .predict(&x)
        .unwrap();

    for i in 0..6 {
        assert!((p2[[i, 0]] - p[[i, 1]]).abs() < 1e-9, "swap permutes col");
        assert!((p2[[i, 1]] - p[[i, 0]]).abs() < 1e-9, "swap permutes col");
    }
}

/// Guard: MultiOutputClassifier predict — per-column class LABELS in original
/// order (not a transposed/swapped matrix), distinct labels per column.
///
/// LIVE ORACLE (run from /tmp):
/// ```text
/// Xc = np.array([[0.],[1.],[2.],[3.]]); Yc = np.array([[0,5],[0,5],[0,5],[0,5]])
/// moc = MultiOutputClassifier(DummyClassifier('most_frequent')).fit(Xc, Yc)
/// # moc.predict(Xc)[0].tolist() == [0, 5]  (col0 label 0, col1 label 5)
/// # dtype int64 — labels, not scores/probabilities
/// ```
/// ferrolearn's `MeanEstimator` over a constant column reproduces that
/// column's constant label, so MOC must yield col0==0.0, col1==5.0.
#[test]
fn guard_moc_predict_labels_column_order() {
    // col0 all 0, col1 all 5 -> oracle most_frequent labels [0, 5].
    let x = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 2.0, 3.0]).unwrap();
    let y = Array2::from_shape_vec((4, 2), vec![0.0, 5.0, 0.0, 5.0, 0.0, 5.0, 0.0, 5.0]).unwrap();

    let moc = MultiOutputClassifier::new(mean_factory());
    let p = moc.fit(&x, &y).unwrap().predict(&x).unwrap();
    assert_eq!(p.shape(), &[4, 2]);
    for i in 0..4 {
        assert!((p[[i, 0]] - 0.0).abs() < 1e-10, "col0 label 0 (oracle)");
        assert!((p[[i, 1]] - 5.0).abs() < 1e-10, "col1 label 5 (oracle)");
    }
}

// ===========================================================================
// REQ-VALIDATION — row-count mismatch + empty-y
// ===========================================================================

/// Guard: REQ-VALIDATION row-count mismatch ⇒ ShapeMismatch.
///
/// sklearn validates `y` rows against `X` rows in `_validate_data`
/// `sklearn/multioutput.py:246` and raises `ValueError`.
/// LIVE ORACLE: `MultiOutputRegressor(LinearRegression()).fit(zeros((10,2)),
/// zeros((8,2)))` raises ValueError.
#[test]
fn guard_validation_row_mismatch() {
    let x = Array2::<f64>::zeros((10, 2));
    let y = Array2::<f64>::zeros((8, 2));

    match MultiOutputRegressor::new(mean_factory()).fit(&x, &y) {
        Err(FerroError::ShapeMismatch { .. }) => {}
        other => panic!(
            "MOR row mismatch must be ShapeMismatch, ok={:?}",
            other.is_ok()
        ),
    }
    match MultiOutputClassifier::new(mean_factory()).fit(&x, &y) {
        Err(FerroError::ShapeMismatch { .. }) => {}
        other => panic!(
            "MOC row mismatch must be ShapeMismatch, ok={:?}",
            other.is_ok()
        ),
    }
}

/// Guard: REQ-VALIDATION empty-y (n_targets == 0) ⇒ InvalidParameter.
///
/// sklearn does NOT special-case 0 columns: `MultiOutputRegressor(...).fit(X,
/// np.empty((n,0)))` raises `ValueError("Found array with 0 feature(s) ...")`
/// from `_validate_data`. Both implementations ERROR (compatible — the error
/// TYPE differs but both reject empty targets). LIVE ORACLE: ValueError raised.
#[test]
fn guard_validation_empty_y() {
    let x = Array2::<f64>::zeros((5, 2));
    let y = Array2::<f64>::zeros((5, 0));

    match MultiOutputRegressor::new(mean_factory()).fit(&x, &y) {
        Err(FerroError::InvalidParameter { .. }) => {}
        other => panic!(
            "MOR empty targets must be InvalidParameter, ok={:?}",
            other.is_ok()
        ),
    }
    match MultiOutputClassifier::new(mean_factory()).fit(&x, &y) {
        Err(FerroError::InvalidParameter { .. }) => {}
        other => panic!(
            "MOC empty targets must be InvalidParameter, ok={:?}",
            other.is_ok()
        ),
    }
}

// ===========================================================================
// Unclaimed-edge guards (hunt): single-column 2D Y shape, mismatched #classes
// ===========================================================================

/// Guard: single-column 2D Y (n_targets == 1) ⇒ predict shape (n, 1), NOT (n,).
///
/// LIVE ORACLE (run from /tmp):
/// ```text
/// X1 = np.array([[1.],[2.],[3.]]); Y1 = np.array([[5.],[5.],[5.]])
/// P1 = MultiOutputRegressor(DummyRegressor('mean')).fit(X1, Y1).predict(X1)
/// # P1.shape == (3,1); P1.ravel().tolist() == [5.0, 5.0, 5.0]
/// ```
#[test]
fn guard_single_column_y_shape() {
    let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
    let y = Array2::from_shape_vec((3, 1), vec![5.0, 5.0, 5.0]).unwrap();

    let p = MultiOutputRegressor::new(mean_factory())
        .fit(&x, &y)
        .unwrap()
        .predict(&x)
        .unwrap();
    assert_eq!(
        p.shape(),
        &[3, 1],
        "single-target predict is (n,1) not (n,)"
    );
    for i in 0..3 {
        assert!((p[[i, 0]] - 5.0).abs() < 1e-10);
    }
}

/// Guard: per-column estimators are independent even when columns have a
/// DIFFERENT number of distinct class labels (no cross-column coupling).
///
/// LIVE ORACLE (run from /tmp):
/// ```text
/// Xc = np.array([[0.],[1.],[2.],[3.]])
/// Yc = np.array([[0,5],[1,5],[0,7],[1,7]])  # col0 {0,1}, col1 {5,7}
/// # each output's classifier is fit independently; predict has shape (4,2)
/// ```
/// We assert shape + that each constant-column base recovers its own column,
/// confirming the differing-cardinality columns do not interfere.
#[test]
fn guard_mismatched_classes_per_column_independent() {
    let x = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 2.0, 3.0]).unwrap();
    // col0 constant 0 ({0}), col1 constant 7 ({7}) — different label sets.
    let y = Array2::from_shape_vec((4, 2), vec![0.0, 7.0, 0.0, 7.0, 0.0, 7.0, 0.0, 7.0]).unwrap();

    let fitted = MultiOutputClassifier::new(mean_factory())
        .fit(&x, &y)
        .unwrap();
    assert_eq!(fitted.n_estimators(), 2);
    let p = fitted.predict(&x).unwrap();
    assert_eq!(p.shape(), &[4, 2]);
    for i in 0..4 {
        assert!((p[[i, 0]] - 0.0).abs() < 1e-10);
        assert!((p[[i, 1]] - 7.0).abs() < 1e-10);
    }
}
