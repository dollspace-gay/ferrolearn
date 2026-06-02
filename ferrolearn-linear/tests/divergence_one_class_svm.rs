//! Divergence pins: `ferrolearn-linear` `OneClassSVM` vs the live scikit-learn
//! 1.5.2 oracle (`sklearn.svm.OneClassSVM`, libsvm `ONE_CLASS`).
//!
//! These tests are authored by the ACToR critic. Each expected value is
//! re-derived LIVE from the installed sklearn 1.5.2 oracle (R-CHAR-3) — never
//! copied from the ferrolearn side. The python invocation that produced each
//! constant is recorded inline above the test.
//!
//! Shared input (per the design doc AC table): the 7x2 set
//! `X = [[0,0],[0.1,0.1],[-0.1,0.1],[0.1,-0.1],[0,0.2],[0.2,0],[3,3]]`
//! whose last row `[3,3]` is the outlier.
//!
//! NOTE (compile gate): `one_class_svm.rs` exposes NO public
//! `dual_coef_`/`support_`/`n_support_`/`intercept_`/`offset_`/`score_samples`/
//! `coef_` accessors. These pins therefore use ONLY the current public API
//! (`fit`, `decision_function`, `predict`, the kernel constructors). The
//! fitted-attribute REQs (#646 dual_coef_, #648 support_/intercept_/offset_/
//! score_samples) are BLOCKED on the builder adding those accessors and are
//! NOT pinned here (they would not compile).

use ferrolearn_core::{Fit, Predict};
use ferrolearn_linear::one_class_svm::OneClassSVM;
use ferrolearn_linear::svm::{LinearKernel, RbfKernel};
use ndarray::{Array1, Array2};

/// The shared 7x2 fixture. Row 6 (`[3,3]`) is the planted outlier.
fn fixture() -> Array2<f64> {
    Array2::from_shape_vec(
        (7, 2),
        vec![
            0.0, 0.0, 0.1, 0.1, -0.1, 0.1, 0.1, -0.1, 0.0, 0.2, 0.2, 0.0, 3.0, 3.0,
        ],
    )
    .unwrap()
}

/// PIN 1 — #646: `decision_function` scaling divergence (the PRIME divergence).
///
/// Divergence: `ferrolearn::one_class_svm::FittedOneClassSVM::decision_function`
/// diverges from `sklearn.svm.OneClassSVM.decision_function`
/// (`sklearn/svm/_classes.py:1770`, `sklearn/svm/src/libsvm/svm.cpp:2834`
/// `sum -= model->rho[0]`).
///
/// ferrolearn solves the NORMALIZED one-class dual (`0<=a<=1/(n*nu)`, `Sum a = 1`,
/// `one_class_svm.rs` `fn fit` lines 164-166) and stores the raw normalized
/// `alpha`/`rho`, while libsvm solves `0<=a<=1`, `Sum a = nu*n` and exposes the
/// un-normalized values. The two optima coincide only up to the scale factor
/// `1/(nu*n)`, so ferrolearn's decision values are `1/(nu*n) = 1/3.5` of libsvm's.
///
/// Oracle (re-derived live):
/// ```text
/// python3 -c "import numpy as np; from sklearn.svm import OneClassSVM; \
/// X=np.array([[0,0],[0.1,0.1],[-0.1,0.1],[0.1,-0.1],[0,0.2],[0.2,0],[3,3]],dtype=float); \
/// m=OneClassSVM(kernel='linear',nu=0.5).fit(X); \
/// print(np.round(m.decision_function(X),6).tolist())"
/// # [-0.01, 0.0, -0.01, -0.01, 0.0, 0.0, 0.29]
/// ```
#[test]
fn divergence_pin1_decision_function_scaling_646() {
    // Expected values come from the LIVE sklearn oracle (recorded above), NOT
    // from ferrolearn (R-CHAR-3).
    let oracle_df: [f64; 7] = [-0.01, 0.0, -0.01, -0.01, 0.0, 0.0, 0.29];

    let x = fixture();
    let model = OneClassSVM::new(LinearKernel)
        .with_nu(0.5)
        .with_tol(1e-9)
        .with_max_iter(1_000_000);
    let fitted = model.fit(&x, &()).expect("fit should succeed");
    let df: Array1<f64> = fitted
        .decision_function(&x)
        .expect("decision_function should succeed");

    // Report ferrolearn vs oracle so the nu*n=3.5 scaling is evident in the log.
    eprintln!("PIN1 oracle df    = {oracle_df:?}");
    eprintln!("PIN1 ferrolearn df= {:?}", df.to_vec());
    if df[6].abs() > 1e-12 {
        eprintln!("PIN1 ratio oracle/ferro at idx6 = {}", oracle_df[6] / df[6]);
    }

    for (i, &exp) in oracle_df.iter().enumerate() {
        assert!(
            (df[i] - exp).abs() < 1e-2,
            "decision_function[{i}]: ferrolearn={} expected (sklearn)={exp} (diff {})",
            df[i],
            (df[i] - exp).abs()
        );
    }
}

/// PIN 2 — #647: `gamma='scale'` default not resolved at fit time.
///
/// Divergence: `sklearn.svm.OneClassSVM(kernel='rbf', nu=0.5)` defaults to
/// `gamma='scale' = 1/(n_features * X.var())` resolved against `X` at fit
/// (`sklearn/svm/_base.py:238-239`). ferrolearn's `one_class_svm.rs` `fn fit`
/// does NOT call `Kernel::resolved_for_fit(x)`, so a default `RbfKernel::new()`
/// (`Gamma::Scale`) silently evaluates with `gamma=1.0`
/// (`svm.rs` `gamma_value_or_one`: `Gamma::Scale | Gamma::Auto => F::one()`).
/// This pin therefore also carries the #646 scaling gap.
///
/// Oracle (re-derived live):
/// ```text
/// python3 -c "import numpy as np; from sklearn.svm import OneClassSVM; \
/// X=np.array([[0,0],[0.1,0.1],[-0.1,0.1],[0.1,-0.1],[0,0.2],[0.2,0],[3,3]],dtype=float); \
/// m=OneClassSVM(kernel='rbf',nu=0.5).fit(X); \
/// print(m._gamma); print(np.round(m.decision_function(X),6).tolist())"
/// # _gamma = 0.46577946768060835   (= 1/(2 * X.var()=1.073469...))
/// # df = [0.022499, 0.022633, 0.000122, 0.0, 0.0, 0.000387, -1.44231]
/// ```
#[test]
fn divergence_pin2_gamma_scale_default_647() {
    // LIVE scale-gamma oracle df (R-CHAR-3).
    let oracle_df: [f64; 7] = [0.022499, 0.022633, 0.000122, 0.0, 0.0, 0.000387, -1.44231];

    let x = fixture();
    // Default RbfKernel (Gamma::Scale). Because fit() never resolves gamma
    // against X, ferrolearn effectively uses gamma=1.0 here.
    let model = OneClassSVM::new(RbfKernel::<f64>::new())
        .with_nu(0.5)
        .with_tol(1e-9)
        .with_max_iter(1_000_000);
    let fitted = model.fit(&x, &()).expect("fit should succeed");
    let df: Array1<f64> = fitted
        .decision_function(&x)
        .expect("decision_function should succeed");

    eprintln!("PIN2 oracle df (scale gamma=0.46578) = {oracle_df:?}");
    eprintln!("PIN2 ferrolearn df (gamma=1.0)        = {:?}", df.to_vec());

    for (i, &exp) in oracle_df.iter().enumerate() {
        assert!(
            (df[i] - exp).abs() < 1e-2,
            "decision_function[{i}]: ferrolearn={} expected (sklearn scale-gamma)={exp} (diff {})",
            df[i],
            (df[i] - exp).abs()
        );
    }
}

/// PIN 3 — #648: `predict` labels (sign convention at the boundary).
///
/// Divergence: `sklearn.svm.OneClassSVM.predict` maps libsvm's
/// `(sum > 0) ? +1 : -1` (`sklearn/svm/src/libsvm/svm.cpp:2837-2838`, strict
/// `>`). ferrolearn's `one_class_svm.rs` `fn predict` uses `val >= F::zero()`
/// (`>=`), so a sample whose decision value is exactly `0` is labelled `+1` by
/// ferrolearn but would be `-1` by libsvm. The predict sign is otherwise
/// scale-invariant under the #646 `nu*n` factor.
///
/// Oracle (re-derived live):
/// ```text
/// python3 -c "import numpy as np; from sklearn.svm import OneClassSVM; \
/// X=np.array([[0,0],[0.1,0.1],[-0.1,0.1],[0.1,-0.1],[0,0.2],[0.2,0],[3,3]],dtype=float); \
/// m=OneClassSVM(kernel='linear',nu=0.5).fit(X); print(m.predict(X).tolist())"
/// # [-1, 1, -1, -1, 1, 1, 1]
/// ```
/// NB: the oracle's labels at indices 1,4,5 are `+1` even though the ROUNDED
/// df reads `0.0` there — libsvm's internal raw `sum` is fractionally positive
/// (`~2.2e-10`), so strict `>0` still gives `+1`. predict therefore MAY pass
/// element-wise on this fixture; the `>=`-vs-`>` boundary divergence bites only
/// at a true exact `0`. We pin the live labels to record the actual status.
#[test]
fn divergence_pin3_predict_labels_648() {
    // LIVE oracle labels (R-CHAR-3).
    let oracle_pred: [isize; 7] = [-1, 1, -1, -1, 1, 1, 1];

    let x = fixture();
    let model = OneClassSVM::new(LinearKernel)
        .with_nu(0.5)
        .with_tol(1e-9)
        .with_max_iter(1_000_000);
    let fitted = model.fit(&x, &()).expect("fit should succeed");
    let pred: Array1<isize> = fitted.predict(&x).expect("predict should succeed");

    eprintln!("PIN3 oracle predict     = {oracle_pred:?}");
    eprintln!("PIN3 ferrolearn predict = {:?}", pred.to_vec());

    for (i, &exp) in oracle_pred.iter().enumerate() {
        assert_eq!(
            pred[i], exp,
            "predict[{i}]: ferrolearn={} expected (sklearn)={exp}",
            pred[i]
        );
    }
}
