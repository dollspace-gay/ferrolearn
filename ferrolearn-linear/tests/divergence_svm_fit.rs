//! Divergence pins for `ferrolearn-linear/src/svm.rs` (SVC/SVR) against the
//! LIVE scikit-learn 1.5.2 oracle.
//!
//! All expected values were re-derived live by the critic from sklearn 1.5.2
//! (R-CHAR-3 — never copied from the ferrolearn side):
//!
//! ```text
//! python3 -c "import numpy as np; from sklearn.svm import SVC, SVR; ..."
//! ```
//!
//! These pins use ONLY the current public API (`SVC::new`/`with_*`, `fit`,
//! `decision_function`, `predict`, the kernel constructors). The libsvm-layout
//! fitted-attribute accessors (`support_`/`dual_coef_`/`n_support_`/
//! `intercept_`/`coef_`) do NOT exist on `FittedSVC`/`FittedSVR`, so REQ-2 /
//! REQ-3 / REQ-6-attrs / REQ-7-per-pair cannot be pinned here without breaking
//! compilation — they are blocked on the builder adding accessors (#636/#639/
//! #640). See the critic report.

use ferrolearn_core::{Fit, Predict};
use ferrolearn_linear::svm::{LinearKernel, RbfKernel, SVC, SVR};
use ndarray::{Array1, Array2, array};

/// Binary 6x2 training set shared by PIN 1 / PIN 2 / PIN 3.
fn binary_6x2() -> (Array2<f64>, Array1<usize>) {
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 5.0, 5.0, 6.0, 5.0, 5.0, 6.0],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 1, 1, 1];
    (x, y)
}

/// PIN 1 — REQ-4: binary `decision_function` shape + sign + values (#637).
///
/// Divergence: `FittedSVC::decision_function` in
/// `ferrolearn-linear/src/svm.rs` returns raw one-vs-one `Array2` shape
/// `(n_samples, n_models)` = `(6, 1)`, whereas sklearn's binary case returns
/// `-dec.ravel()` shape `(n_samples,)` = `(6,)`
/// (`sklearn/svm/_base.py:538-539`: `return -dec_func.ravel()`).
///
/// Live oracle `SVC(kernel='linear', C=1.0)` on the binary 6x2 set:
///   df.shape == (6,)
///   df == [-1.285333, -0.999733, -0.999733, 0.999467, 1.285067, 1.285067]
/// (positive -> classes_[1]).
///
/// This pin asserts ferrolearn's binary decision values match the oracle's
/// 1-D `(6,)` values element-wise (reading whatever ferrolearn exposes — here
/// column 0 of its `(6,1)` Array2). It FAILS on the value mismatch (and the
/// conceptual shape divergence is documented): ferrolearn's SMO does not
/// converge to libsvm's optimum / its bias-recovery differs, so the per-sample
/// margins do not equal the oracle's.
///
/// Tracking: #637
#[test]
fn divergence_pin1_binary_decision_function_values() {
    let (x, y) = binary_6x2();
    let model = SVC::<f64, LinearKernel>::new(LinearKernel)
        .with_c(1.0)
        .with_tol(1e-6)
        .with_max_iter(1_000_000);
    let fitted = model.fit(&x, &y).unwrap();
    let df = fitted.decision_function(&x).unwrap();

    // sklearn binary decision_function is conceptually 1-D shape (6,).
    // ferrolearn returns Array2 (6,1); read its single column.
    assert_eq!(df.nrows(), 6, "expected 6 rows");

    // Live oracle values from SVC(kernel='linear',C=1.0).decision_function(X).
    let oracle: [f64; 6] = [
        -1.285333, -0.999733, -0.999733, 0.999467, 1.285067, 1.285067,
    ];
    for (i, &exp) in oracle.iter().enumerate() {
        let got = df[[i, 0]];
        assert!(
            (got - exp).abs() < 1e-2,
            "binary decision_function[{i}]: ferrolearn={got}, sklearn={exp}, gap={}",
            (got - exp).abs()
        );
    }
}

/// PIN 2 — REQ-1: default `gamma='scale'` divergence (#634).
///
/// Divergence: sklearn `SVC()` defaults to `kernel='rbf', gamma='scale'`,
/// resolving `_gamma = 1/(n_features * X.var())`
/// (`sklearn/svm/_base.py:236-239`). On the binary 6x2 set the live oracle
/// gives `_gamma == 0.11842105263157894` and
///   decision_function == [-1.013804, -1.000632, -1.000319,
///                          1.000308, 1.000335, 1.000308].
///
/// ferrolearn's `RbfKernel::new()` (gamma = None) resolves a `None` gamma to
/// `F::one()` (= 1.0) in `RbfKernel::compute` — NOT 0.1184 — and the kernel
/// has no access to X, so `gamma='scale'` is unimplemented. A default-gamma
/// `RbfKernel` SVC therefore diverges from the oracle's scale-gamma fit.
///
/// Tracking: #634
#[test]
fn divergence_pin2_rbf_default_scale_gamma() {
    let (x, y) = binary_6x2();
    // Default RBF kernel: gamma=None -> ferrolearn uses 1.0 (NOT 0.1184 scale).
    let model = SVC::new(RbfKernel::<f64>::new())
        .with_c(1.0)
        .with_tol(1e-6)
        .with_max_iter(1_000_000);
    let fitted = model.fit(&x, &y).unwrap();
    let df = fitted.decision_function(&x).unwrap();

    // Live oracle SVC(C=1.0) (default rbf, gamma='scale'=0.1184) values.
    let oracle: [f64; 6] = [
        -1.013804, -1.000632, -1.000319, 1.000308, 1.000335, 1.000308,
    ];
    for (i, &exp) in oracle.iter().enumerate() {
        let got = df[[i, 0]];
        assert!(
            (got - exp).abs() < 1e-2,
            "default-rbf decision_function[{i}]: ferrolearn(gamma=1.0)={got}, \
             sklearn(gamma=scale=0.1184)={exp}, gap={}",
            (got - exp).abs()
        );
    }
}

/// PIN 3 — REQ-5: `predict` labels (#638).
///
/// Divergence target: ferrolearn `SVC(LinearKernel)` `predict` must equal the
/// live oracle `SVC(kernel='linear', C=1.0).predict(X)` labels element-wise,
/// on both the binary 6x2 set and a 3-class 9x2 linear set.
///
/// Live oracle:
///   binary 6x2  predict == [0, 0, 0, 1, 1, 1]
///   3-class 9x2 predict == [0, 0, 0, 1, 1, 1, 2, 2, 2]
///
/// (predict may already pass when the fit recovers the right margins; a green
/// pin certifies REQ-5's voting structure, a red one is a real divergence —
/// the critic report records which.)
///
/// Tracking: #638
#[test]
fn divergence_pin3_predict_labels() {
    // Binary.
    let (xb, yb) = binary_6x2();
    let mb = SVC::<f64, LinearKernel>::new(LinearKernel)
        .with_c(1.0)
        .with_tol(1e-6)
        .with_max_iter(1_000_000);
    let fb = mb.fit(&xb, &yb).unwrap();
    let pb = fb.predict(&xb).unwrap();
    let oracle_b = [0usize, 0, 0, 1, 1, 1];
    for (i, &exp) in oracle_b.iter().enumerate() {
        assert_eq!(
            pb[i], exp,
            "binary predict[{i}]: ferrolearn={}, sklearn={exp}",
            pb[i]
        );
    }

    // 3-class.
    let x3 = Array2::from_shape_vec(
        (9, 2),
        vec![
            0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 5.0, 0.0, 5.5, 0.0, 5.0, 0.5, 0.0, 5.0, 0.5, 5.0, 0.0,
            5.5,
        ],
    )
    .unwrap();
    let y3 = array![0usize, 0, 0, 1, 1, 1, 2, 2, 2];
    let m3 = SVC::<f64, LinearKernel>::new(LinearKernel)
        .with_c(1.0)
        .with_tol(1e-6)
        .with_max_iter(1_000_000);
    let f3 = m3.fit(&x3, &y3).unwrap();
    let p3 = f3.predict(&x3).unwrap();
    let oracle_3 = [0usize, 0, 0, 1, 1, 1, 2, 2, 2];
    for (i, &exp) in oracle_3.iter().enumerate() {
        assert_eq!(
            p3[i], exp,
            "3-class predict[{i}]: ferrolearn={}, sklearn={exp}",
            p3[i]
        );
    }
}

/// PIN 4 — REQ-6: SVR `predict` values (#639).
///
/// Divergence target: ferrolearn `SVR(LinearKernel, C=100, epsilon=0.1)`
/// `predict` must equal the live oracle
/// `SVR(kernel='linear', C=100, epsilon=0.1).predict(X)` on the 6x1 set
/// `X=[[1],[2],[3],[4],[5],[6]]`, `y=[2,4,6,8,10,12]`.
///
/// Live oracle predict == [2.1, 4.06, 6.02, 7.98, 9.94, 11.9].
///
/// Tracking: #639
#[test]
fn divergence_pin4_svr_predict_values() {
    let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let y = array![2.0f64, 4.0, 6.0, 8.0, 10.0, 12.0];
    let model = SVR::<f64, LinearKernel>::new(LinearKernel)
        .with_c(100.0)
        .with_epsilon(0.1)
        .with_tol(1e-6)
        .with_max_iter(1_000_000);
    let fitted = model.fit(&x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();

    // Live oracle SVR(kernel='linear',C=100,epsilon=0.1).predict(X).
    let oracle: [f64; 6] = [2.1, 4.06, 6.02, 7.98, 9.94, 11.9];
    for (i, &exp) in oracle.iter().enumerate() {
        assert!(
            (preds[i] - exp).abs() < 1e-2,
            "SVR predict[{i}]: ferrolearn={}, sklearn={exp}, gap={}",
            preds[i],
            (preds[i] - exp).abs()
        );
    }
}
