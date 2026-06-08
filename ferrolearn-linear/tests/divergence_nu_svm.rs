//! Divergence pins for `ferrolearn-linear/src/nu_svm.rs` (`NuSVC` / `NuSVR`)
//! against the LIVE scikit-learn 1.5.2 oracle.
//!
//! `NuSVC`/`NuSVR` are libsvm *nu-parameterized* SVMs (`_impl = "nu_svc"`,
//! `solver_type == 1`; `_impl = "nu_svr"`, `solver_type == 4` —
//! `sklearn/svm/_base.py:30`). ferrolearn now runs the GENUINE libsvm nu-SVM
//! solver (`crate::svm::solve_nu_svc`/`solve_nu_svr`, built on the shared
//! `solver_nu_core` = libsvm `Solver_NU`), NOT a C-SVC / epsilon-SVR with a
//! re-scaled `C`. These pins certify the rewired solver: each compares
//! `import ferrolearn` against the live `sklearn` 1.5.2 oracle.
//!
//! Expected values are computed LIVE from sklearn 1.5.2 (R-CHAR-3); the python
//! invocations are recorded inline above each pin. Tolerance 1e-2 for values,
//! exact for `support_` indices and `predict` labels.
//!
//! The first four pins (`divergence_nusvc_decision_function_delegation`,
//! `nusvc_predict_labels_match_on_separable`, `divergence_nusvr_predict_delegation`,
//! `divergence_nusvr_missing_c_parameter`) were authored when ferrolearn still
//! delegated to C-SVC / epsilon-SVR; after the genuine-`Solver_NU` rewire they
//! now CERTIFY the correct nu-SVM optimum (all GREEN). Pins A/B/C below
//! additionally certify the RE-EXPOSED fitted-attribute accessors on
//! `FittedNuSVC`/`FittedNuSVR` (#657) and the multiclass ovo wiring.

use ferrolearn_core::{Fit, Predict};
use ferrolearn_linear::nu_svm::{NuSVC, NuSVR};
use ferrolearn_linear::svm::{LinearKernel, RbfKernel};
use ndarray::{Array1, Array2, array};

/// Divergence: `NuSVC::decision_function` diverges from
/// `sklearn.svm.NuSVC` (libsvm `solve_nu_svc`, `sklearn/svm/src/libsvm/svm.cpp`;
/// `_impl = "nu_svc"` → `solver_type == 1` at `sklearn/svm/_base.py:30`).
///
/// Live oracle (sklearn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; from sklearn.svm import NuSVC; \
///   X=np.array([[1.,1.],[2.,1.],[1.,2.],[5.,5.],[6.,5.],[5.,6.]]); y=np.array([0,0,0,1,1,1]); \
///   m=NuSVC(kernel='linear',nu=0.5).fit(X,y); print(np.round(m.decision_function(X),6).tolist())"
/// # -> [-1.25, -1.0, -1.0, 0.75, 1.0, 1.0]   predict -> [0,0,0,1,1,1]
/// ```
/// Post-rewire this is GREEN: ferrolearn's genuine nu-SVC solver reaches the
/// libsvm optimum, not the C-SVC delegation optimum.
/// Tracking: #<filed-below>
#[test]
fn divergence_nusvc_decision_function_delegation() {
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 5.0, 5.0, 6.0, 5.0, 5.0, 6.0],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 1, 1, 1];

    // Live sklearn NuSVC(kernel='linear', nu=0.5) oracle (NOT copied from ferrolearn).
    let oracle_df: [f64; 6] = [-1.25, -1.0, -1.0, 0.75, 1.0, 1.0];

    let model = NuSVC::<f64, LinearKernel>::new(LinearKernel)
        .with_nu(0.5)
        .with_max_iter(200_000)
        .with_tol(1e-6);
    let fitted = model.fit(&x, &y).unwrap();

    let scores = fitted.decision_function(&x).unwrap();
    let df = scores
        .as_binary()
        .expect("binary NuSVC decision_function must be SvmScores::Binary");

    // Report ferrolearn's actual values for the audit trail.
    let ferro: Vec<f64> = df.iter().copied().collect();
    eprintln!("nusvc df  ferrolearn = {ferro:?}");
    eprintln!("nusvc df  oracle     = {oracle_df:?}");

    for (i, (&f, &o)) in df.iter().zip(oracle_df.iter()).enumerate() {
        assert!(
            (f - o).abs() < 1e-2,
            "NuSVC decision_function[{i}] ferrolearn={f} vs sklearn nu-SVC oracle={o} \
             (genuine solve_nu_svc must reach the libsvm optimum; \
             sklearn/svm/src/libsvm/svm.cpp solve_nu_svc)"
        );
    }
}

/// `NuSVC::predict` labels vs `sklearn.svm.NuSVC`.
///
/// Live oracle:
/// ```text
/// python3 -c "...; print(NuSVC(kernel='linear',nu=0.5).fit(X,y).predict(X).tolist())"
/// # -> [0, 0, 0, 1, 1, 1]
/// ```
#[test]
fn nusvc_predict_labels_match_on_separable() {
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 5.0, 5.0, 6.0, 5.0, 5.0, 6.0],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 1, 1, 1];

    let oracle_labels = array![0usize, 0, 0, 1, 1, 1];

    let model = NuSVC::<f64, LinearKernel>::new(LinearKernel)
        .with_nu(0.5)
        .with_max_iter(200_000)
        .with_tol(1e-6);
    let fitted = model.fit(&x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();

    assert_eq!(
        preds, oracle_labels,
        "NuSVC predict labels diverge from sklearn nu-SVC oracle"
    );
}

/// Divergence: `NuSVR::predict` diverges from `sklearn.svm.NuSVR` (libsvm
/// `solve_nu_svr`, `sklearn/svm/src/libsvm/svm.cpp`; `_impl = "nu_svr"` →
/// `solver_type == 4` at `sklearn/svm/_base.py:30`).
///
/// Live oracle (sklearn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; from sklearn.svm import NuSVR; \
///   X=np.array([1.,2.,3.,4.]).reshape(-1,1); y=np.array([1.,5.,2.,8.]); \
///   print(np.round(NuSVR(kernel='linear',nu=0.5,C=1.0).fit(X,y).predict(X),6).tolist())"
/// # NuSVR(C=1.0) -> [2.5, 3.5, 4.5, 5.5]   support_ [2,3] dual_coef_ [[-1,1]] intercept_ [1.5]
/// ```
/// Post-rewire GREEN: ferrolearn's `NuSVR(nu=0.5, C=1.0)` reaches the libsvm
/// nu-SVR optimum.
/// Tracking: #<filed-below>
#[test]
fn divergence_nusvr_predict_delegation() {
    let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let y: Array1<f64> = array![1.0, 5.0, 2.0, 8.0];

    // Live sklearn NuSVR(kernel='linear', nu=0.5, C=1.0) oracle.
    let oracle_pred: [f64; 4] = [2.5, 3.5, 4.5, 5.5];

    let model = NuSVR::<f64, LinearKernel>::new(LinearKernel)
        .with_nu(0.5)
        .with_c(1.0)
        .with_max_iter(500_000)
        .with_tol(1e-7);
    let fitted = model.fit(&x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();

    let ferro: Vec<f64> = preds.iter().copied().collect();
    eprintln!("nusvr pred ferrolearn = {ferro:?}");
    eprintln!("nusvr pred oracle     = {oracle_pred:?}");

    for (i, (&p, &o)) in preds.iter().zip(oracle_pred.iter()).enumerate() {
        assert!(
            (p - o).abs() < 1e-2,
            "NuSVR predict[{i}] ferrolearn={p} vs sklearn nu-SVR(C=1.0) oracle={o} \
             (genuine solve_nu_svr must reach the libsvm optimum; \
             sklearn/svm/src/libsvm/svm.cpp solve_nu_svr)"
        );
    }
}

/// `sklearn.svm.NuSVR` exposes a `C` parameter (default 1.0,
/// `sklearn/svm/_classes.py:1531`). ferrolearn's `NuSVR` now carries `pub c: F`
/// (default 1.0) + `with_c`. This pin asserts `NuSVR(nu=0.5, C=1.0)`
/// reproduces the sklearn DEFAULT `NuSVR(nu=0.5, C=1.0)` predictions.
///
/// Live oracle: NuSVR(nu=0.5, C=1.0) predict -> [2.5, 3.5, 4.5, 5.5].
/// Tracking: #<filed-below>
#[test]
fn divergence_nusvr_missing_c_parameter() {
    let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let y: Array1<f64> = array![1.0, 5.0, 2.0, 8.0];

    // sklearn NuSVR's default C is 1.0; the oracle prediction with that default.
    let oracle_pred_c_default: [f64; 4] = [2.5, 3.5, 4.5, 5.5];

    let model = NuSVR::<f64, LinearKernel>::new(LinearKernel)
        .with_nu(0.5)
        .with_c(1.0)
        .with_max_iter(500_000)
        .with_tol(1e-7);
    let fitted = model.fit(&x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();

    for (i, (&p, &o)) in preds.iter().zip(oracle_pred_c_default.iter()).enumerate() {
        assert!(
            (p - o).abs() < 1e-2,
            "NuSVR predict[{i}] ferrolearn={p} vs sklearn NuSVR(C=1.0 default) oracle={o}: \
             (sklearn/svm/_classes.py:1531 default C=1.0)"
        );
    }
}

// ===========================================================================
// PIN A — #657: re-exposed FittedNuSVC fitted attributes (support_,
// n_support_, dual_coef_, intercept_, coef_) vs the live NuSVC oracle.
// ===========================================================================

/// Certifies the RE-EXPOSED `FittedNuSVC::{support,n_support,dual_coef,
/// intercept,coef}` accessors (#657) against the live `sklearn.svm.NuSVC`
/// libsvm-layout fitted attributes (`sklearn/svm/_base.py:318-410` `support_`,
/// `:258-262` the nu_svc binary sign flip, `:650-666` linear `coef_`).
///
/// Live oracle (sklearn 1.5.2, R-CHAR-3 — values from sklearn, never copied
/// from ferrolearn):
/// ```text
/// python3 -c "import numpy as np; from sklearn.svm import NuSVC; \
///   X=np.array([[1.,1.],[2.,1.],[1.,2.],[5.,5.],[6.,5.],[5.,6.]]); y=np.array([0,0,0,1,1,1]); \
///   m=NuSVC(kernel='linear',nu=0.5).fit(X,y); \
///   print(m.support_.tolist(), m.n_support_.tolist(), \
///         np.round(m.dual_coef_,6).tolist(), np.round(m.intercept_,6).tolist(), \
///         np.round(m.coef_,6).tolist())"
/// # support_      [1, 2, 3, 5]
/// # n_support_    [2, 2]
/// # dual_coef_    [[-0.022727, -0.045455, 0.045455, 0.022727]]
/// # intercept_    [-1.75]
/// # coef_         [[0.25, 0.25]]
/// # NuSVC(kernel='rbf') => m.coef_ raises AttributeError (linear-only)
/// ```
/// Tracking: #<filed-below>
#[test]
fn nusvc_fitted_attrs_match_oracle() {
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 5.0, 5.0, 6.0, 5.0, 5.0, 6.0],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 1, 1, 1];

    // Live sklearn NuSVC(kernel='linear', nu=0.5) oracle.
    let oracle_support: [usize; 4] = [1, 2, 3, 5];
    let oracle_n_support: [usize; 2] = [2, 2];
    let oracle_dual_coef: [f64; 4] = [-0.022727, -0.045455, 0.045455, 0.022727];
    let oracle_intercept = -1.75_f64;
    let oracle_coef: [f64; 2] = [0.25, 0.25];

    let model = NuSVC::<f64, LinearKernel>::new(LinearKernel)
        .with_nu(0.5)
        .with_max_iter(200_000)
        .with_tol(1e-7);
    let fitted = model.fit(&x, &y).unwrap();

    // support_ — exact (R-CHAR-3: indices exact).
    let support: Vec<usize> = fitted.support().to_vec();
    eprintln!("nusvc support ferrolearn={support:?} oracle={oracle_support:?}");
    assert_eq!(
        support,
        oracle_support.to_vec(),
        "support_ mismatch vs sklearn NuSVC"
    );

    // n_support_ — exact.
    assert_eq!(
        fitted.n_support(),
        oracle_n_support.to_vec(),
        "n_support_ mismatch vs sklearn NuSVC"
    );

    // dual_coef_ — element-wise within 1e-2 in support order.
    let dc = fitted.dual_coef();
    assert_eq!(dc.dim(), (1, 4), "dual_coef_ shape vs sklearn NuSVC");
    for (k, &o) in oracle_dual_coef.iter().enumerate() {
        assert!(
            (dc[[0, k]] - o).abs() < 1e-2,
            "dual_coef_[{k}] ferrolearn={} vs sklearn oracle={o}",
            dc[[0, k]]
        );
    }

    // intercept_ — within 1e-2.
    let ic = fitted.intercept();
    assert!(
        (ic[0] - oracle_intercept).abs() < 1e-2,
        "intercept_ ferrolearn={} vs sklearn oracle={oracle_intercept}",
        ic[0]
    );

    // coef_ (linear) — Some, within 1e-2.
    let coef = fitted
        .coef()
        .expect("linear NuSVC coef_ must be Some (sklearn/svm/_base.py:650-666)");
    assert_eq!(coef.dim(), (1, 2), "coef_ shape vs sklearn NuSVC");
    for (k, &o) in oracle_coef.iter().enumerate() {
        assert!(
            (coef[[0, k]] - o).abs() < 1e-2,
            "coef_[{k}] ferrolearn={} vs sklearn oracle={o}",
            coef[[0, k]]
        );
    }

    // coef_ is None for a non-linear kernel (sklearn raises AttributeError;
    // ferrolearn's contract returns None — `sklearn/svm/_base.py:650-655`).
    let model_rbf = NuSVC::new(RbfKernel::with_gamma(0.5)).with_nu(0.5);
    let fitted_rbf = model_rbf.fit(&x, &y).unwrap();
    assert!(
        fitted_rbf.coef().is_none(),
        "coef_ must be None for an RBF-kernel NuSVC (linear-only attr)"
    );
}

// ===========================================================================
// PIN B — #657: re-exposed FittedNuSVR fitted attributes (support_,
// dual_coef_, intercept_) + predict vs the live NuSVR oracle.
// ===========================================================================

/// Certifies the RE-EXPOSED `FittedNuSVR::{support,dual_coef,intercept}`
/// accessors (#657) + `predict` against the live `sklearn.svm.NuSVR`
/// (`_impl = "nu_svr"`, `solver_type == 4`).
///
/// Live oracle (sklearn 1.5.2, R-CHAR-3):
/// ```text
/// python3 -c "import numpy as np; from sklearn.svm import NuSVR; \
///   X=np.array([[1.],[2.],[3.],[4.]]); y=np.array([1.,5.,2.,8.]); \
///   m=NuSVR(kernel='linear',nu=0.5,C=1.0).fit(X,y); \
///   print(m.support_.tolist(), np.round(m.dual_coef_,6).tolist(), \
///         np.round(m.intercept_,6).tolist(), np.round(m.predict(X),6).tolist())"
/// # support_   [2, 3]
/// # dual_coef_ [[-1.0, 1.0]]
/// # intercept_ [1.5]
/// # predict    [2.5, 3.5, 4.5, 5.5]
/// ```
/// Tracking: #<filed-below>
#[test]
fn nusvr_fitted_attrs_match_oracle() {
    let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let y: Array1<f64> = array![1.0, 5.0, 2.0, 8.0];

    // Live sklearn NuSVR(kernel='linear', nu=0.5, C=1.0) oracle.
    let oracle_support: [usize; 2] = [2, 3];
    let oracle_dual_coef: [f64; 2] = [-1.0, 1.0];
    let oracle_intercept = 1.5_f64;
    let oracle_pred: [f64; 4] = [2.5, 3.5, 4.5, 5.5];

    let model = NuSVR::<f64, LinearKernel>::new(LinearKernel)
        .with_nu(0.5)
        .with_c(1.0)
        .with_max_iter(500_000)
        .with_tol(1e-8);
    let fitted = model.fit(&x, &y).unwrap();

    // support_ — exact.
    let support: Vec<usize> = fitted.support().to_vec();
    eprintln!("nusvr support ferrolearn={support:?} oracle={oracle_support:?}");
    assert_eq!(
        support,
        oracle_support.to_vec(),
        "support_ mismatch vs sklearn NuSVR"
    );

    // dual_coef_ — within 1e-2.
    let dc = fitted.dual_coef();
    assert_eq!(dc.dim(), (1, 2), "dual_coef_ shape vs sklearn NuSVR");
    for (k, &o) in oracle_dual_coef.iter().enumerate() {
        assert!(
            (dc[[0, k]] - o).abs() < 1e-2,
            "dual_coef_[{k}] ferrolearn={} vs sklearn oracle={o}",
            dc[[0, k]]
        );
    }

    // intercept_ — within 1e-2.
    let ic = fitted.intercept();
    assert!(
        (ic[0] - oracle_intercept).abs() < 1e-2,
        "intercept_ ferrolearn={} vs sklearn oracle={oracle_intercept}",
        ic[0]
    );

    // predict — within 1e-2.
    let preds = fitted.predict(&x).unwrap();
    for (i, &o) in oracle_pred.iter().enumerate() {
        assert!(
            (preds[i] - o).abs() < 1e-2,
            "predict[{i}] ferrolearn={} vs sklearn oracle={o}",
            preds[i]
        );
    }
}

// ===========================================================================
// PIN C — multiclass NuSVC (ovo, the true nu-SVC solver per pair) vs the live
// 3-class NuSVC oracle: decision_function shape (n,3) ovr, predict labels,
// dual_coef_ shape (2, n_SV).
// ===========================================================================

/// Certifies the ovo wiring uses the GENUINE nu-SVC per pair on a 3-class
/// linear set, matching `sklearn.svm.NuSVC(kernel='linear', nu=0.3)` for the
/// ovr `decision_function` shape `(n, 3)` (`sklearn/svm/_base.py:778-781`),
/// the `predict` labels (libsvm ovo voting, `:813-814`), and the public
/// `dual_coef_` shape `(n_class-1, n_SV) = (2, n_SV)` (`:258-262`).
///
/// nu=0.3 is feasible for all three class pairs (4 samples/class); sklearn
/// fits without raising.
///
/// Live oracle (sklearn 1.5.2, R-CHAR-3):
/// ```text
/// python3 -c "import numpy as np; from sklearn.svm import NuSVC; \
///   X=np.array([[0.,0.],[0.5,0.],[0.,0.5],[0.5,0.5], \
///               [5.,5.],[5.5,5.],[5.,5.5],[5.5,5.5], \
///               [0.,5.],[0.5,5.],[0.,5.5],[0.5,5.5]]); \
///   y=np.array([0,0,0,0,1,1,1,1,2,2,2,2]); \
///   m=NuSVC(kernel='linear',nu=0.3).fit(X,y); \
///   print(m.classes_.tolist(), m.decision_function(X).shape, \
///         m.predict(X).tolist(), m.dual_coef_.shape)"
/// # classes_       [0, 1, 2]
/// # df shape       (12, 3)
/// # predict        [0,0,0,0,1,1,1,1,2,2,2,2]
/// # dual_coef_     (2, 9)        n_support_ [3, 3, 3]
/// ```
/// Tracking: #<filed-below>
#[test]
fn nusvc_multiclass_ovo_matches_oracle() {
    #[rustfmt::skip]
    let x = Array2::from_shape_vec(
        (12, 2),
        vec![
            0.0, 0.0,  0.5, 0.0,  0.0, 0.5,  0.5, 0.5,
            5.0, 5.0,  5.5, 5.0,  5.0, 5.5,  5.5, 5.5,
            0.0, 5.0,  0.5, 5.0,  0.0, 5.5,  0.5, 5.5,
        ],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];

    // Live sklearn NuSVC(kernel='linear', nu=0.3) oracle.
    let oracle_predict = array![0usize, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];

    let model = NuSVC::<f64, LinearKernel>::new(LinearKernel)
        .with_nu(0.3)
        .with_max_iter(500_000)
        .with_tol(1e-7);
    let fitted = model.fit(&x, &y).unwrap();

    // decision_function ovr shape (12, 3).
    let scores = fitted.decision_function(&x).unwrap();
    let df = scores
        .as_multiclass()
        .expect("3-class ovr NuSVC decision_function must be SvmScores::Multiclass");
    eprintln!(
        "nusvc multiclass df shape ferrolearn={:?} oracle=(12, 3)",
        df.dim()
    );
    assert_eq!(
        df.dim(),
        (12, 3),
        "ovr decision_function shape mismatch vs sklearn NuSVC (sklearn/svm/_base.py:778-781)"
    );

    // predict labels — exact (libsvm ovo voting).
    let preds = fitted.predict(&x).unwrap();
    eprintln!("nusvc multiclass predict ferrolearn={preds:?} oracle={oracle_predict:?}");
    assert_eq!(
        preds, oracle_predict,
        "multiclass predict labels diverge from sklearn 3-class NuSVC ovo oracle"
    );

    // dual_coef_ public layout shape (n_class-1, n_SV) = (2, n_SV).
    let dc = fitted.dual_coef();
    eprintln!(
        "nusvc multiclass dual_coef_ shape ferrolearn={:?}",
        dc.dim()
    );
    assert_eq!(
        dc.dim().0,
        2,
        "dual_coef_ must have n_class-1 = 2 rows for 3-class NuSVC (sklearn/svm/_base.py:258-262)"
    );

    // Argmax of the ovr decision_function must equal predict (structural ovr
    // property the oracle satisfies); certifies the ovr transform is wired
    // through the genuine per-pair nu-SVC votes, not a stub.
    for i in 0..12 {
        let row = df.row(i);
        let mut best = 0usize;
        for c in 1..3 {
            if row[c] > row[best] {
                best = c;
            }
        }
        assert_eq!(
            best,
            preds[i],
            "ovr argmax row {i} ({:?}) disagrees with predict {} (sklearn ovr contract)",
            row.to_vec(),
            preds[i]
        );
    }
}

// ===========================================================================
// REQ-9: probability / predict_proba (NuSVC, #653).
//
// `NuSVC(probability=True)` exposes `predict_proba` via the Platt-scaling
// internal 5-fold CV trained with the NU-SVC sub-solver (the SAME `svm_type`
// as the outer model, libsvm `svm_binary_svc_probability` `svm.cpp:2147-2150`,
// `_impl in ("c_svc","nu_svc")` `_base.py:825`).
//
// The predict_proba VALUES are NOT bit-matchable against sklearn: libsvm seeds
// the CV fold permutation with `random_state` (`svm.cpp:2116-2122`), so
// sklearn's `probA_`/`probB_`/`predict_proba` are NON-DETERMINISTIC across
// `random_state` (the same documented boundary SVC's REQ-9 carries, R-DEV-4).
// These pins therefore assert only sklearn's DOCUMENTED STRUCTURAL CONTRACT
// (R-CHAR-3 — invariants, NOT copied ferrolearn values):
//   - predict_proba shape (n, n_classes); columns = classes_ in sorted order
//     (`_base.py:844-847`);
//   - every row is a probability distribution (sums to 1, values in [0,1]);
//   - binary P(classes_[1]) monotone non-decreasing in the decision value
//     (the sigmoid `1/(1+exp(A f + B))` contract);
//   - predict_log_proba == log(predict_proba) (`_base.py:894`);
//   - probability=False -> predict_proba raises (`_base.py:822-823`/856-860).
// Live oracle confirmation (sklearn 1.5.2): on the binary 10x2 set below,
//   NuSVC(kernel='linear',nu=0.5,probability=True,random_state=0)
//   .predict_proba(X).sum(1) == [1.,...,1.]; P(class1) rises monotonically with
//   the decision value; NuSVC(probability=False).predict_proba raises.
// ===========================================================================

/// Binary 10x2 fit with `probability=True` (linear nu-SVC), nu=0.5.
fn nusvc_proba_binary_fit() -> ferrolearn_linear::nu_svm::FittedNuSVC<f64, LinearKernel> {
    let x = Array2::from_shape_vec(
        (10, 2),
        vec![
            1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.5, 1.5, 5.0, 5.0, 6.0, 5.0, 5.0, 6.0, 6.0,
            6.0, 5.5, 5.5,
        ],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 0, 0, 1, 1, 1, 1, 1];
    NuSVC::<f64, LinearKernel>::new(LinearKernel)
        .with_nu(0.5)
        .with_tol(1e-6)
        .with_max_iter(200_000)
        .with_probability(true)
        .fit(&x, &y)
        .unwrap()
}

#[test]
fn nusvc_predict_proba_raises_when_probability_false() {
    // probability=false (default): predict_proba/predict_log_proba error,
    // mirroring sklearn's raise (`_base.py:822-823`/856-860). Live oracle:
    // NuSVC(probability=False).predict_proba -> AttributeError.
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 5.0, 5.0, 6.0, 5.0, 5.0, 6.0],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 1, 1, 1];
    let fitted = NuSVC::<f64, LinearKernel>::new(LinearKernel)
        .with_nu(0.5)
        .fit(&x, &y)
        .unwrap();
    let q = Array2::from_shape_vec((1, 2), vec![3.0, 3.0]).unwrap();
    assert!(
        fitted.predict_proba(&q).is_err(),
        "predict_proba must raise when probability=false (sklearn _base.py:822)"
    );
    assert!(
        fitted.predict_log_proba(&q).is_err(),
        "predict_log_proba must raise when probability=false"
    );
}

#[test]
fn nusvc_predict_proba_binary_rows_sum_to_one() {
    // sklearn contract (`_base.py:844-847`): predict_proba is (n, n_classes),
    // columns = classes_ sorted; each row a probability distribution.
    let fitted = nusvc_proba_binary_fit();
    let x = Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.5, 1.5, 5.0, 5.0, 5.5, 5.5]).unwrap();
    let p = fitted.predict_proba(&x).unwrap();
    assert_eq!(
        p.dim(),
        (4, 2),
        "predict_proba shape must be (n, n_classes)"
    );
    for s in 0..4 {
        let row_sum = p[[s, 0]] + p[[s, 1]];
        assert!(
            (row_sum - 1.0).abs() < 1e-9,
            "row {s} sums to {row_sum}, must be 1 (probability distribution)"
        );
        for c in 0..2 {
            assert!(
                p[[s, c]] >= 0.0 && p[[s, c]] <= 1.0,
                "p[{s},{c}] = {} out of [0,1]",
                p[[s, c]]
            );
        }
    }
}

#[test]
fn nusvc_predict_proba_binary_monotone_in_decision() {
    // STRUCTURAL invariant (sklearn's sigmoid contract `1/(1+exp(A f + B))`):
    // P(classes_[1]) is monotone non-decreasing in the binary decision value.
    // Confirmed live: sklearn NuSVC(probability=True) P(class1) rises with df.
    let fitted = nusvc_proba_binary_fit();
    let x = Array2::from_shape_vec(
        (5, 2),
        vec![1.0, 1.0, 2.5, 2.5, 3.5, 3.5, 4.5, 4.5, 6.0, 6.0],
    )
    .unwrap();
    let p = fitted.predict_proba(&x).unwrap();
    let df = fitted.decision_function(&x).unwrap();
    let bin = df.as_binary().expect("binary scores");

    let mut order: Vec<usize> = (0..5).collect();
    order.sort_by(|&a, &b| {
        bin[a]
            .partial_cmp(&bin[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut prev = f64::NEG_INFINITY;
    for &s in &order {
        let p1 = p[[s, 1]];
        assert!(
            p1 >= prev - 1e-9,
            "P(class_1) not monotone in decision: sample {s} df={} p1={p1} prev={prev}",
            bin[s]
        );
        prev = p1;
    }
}

#[test]
fn nusvc_predict_log_proba_equals_log_of_proba() {
    // sklearn `predict_log_proba = np.log(predict_proba)` (`_base.py:894`).
    let fitted = nusvc_proba_binary_fit();
    let x = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 3.5, 3.5, 6.0, 6.0]).unwrap();
    let p = fitted.predict_proba(&x).unwrap();
    let lp = fitted.predict_log_proba(&x).unwrap();
    assert_eq!(lp.dim(), p.dim());
    for s in 0..3 {
        for c in 0..2 {
            assert!(
                (lp[[s, c]] - p[[s, c]].ln()).abs() < 1e-12,
                "log_proba[{s},{c}] = {} != ln(proba) {}",
                lp[[s, c]],
                p[[s, c]].ln()
            );
        }
    }
}

#[test]
fn nusvc_predict_proba_multiclass_rows_sum_to_one() {
    // 3-class: predict_proba is (n, 3), each row a probability distribution
    // (Wu-Lin-Weng coupling). Live oracle: sklearn NuSVC 3-class predict_proba
    // rows sum to 1; probA_ has 3 entries (n_pairs).
    let x = Array2::from_shape_vec(
        (9, 2),
        vec![
            0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 5.0, 0.0, 5.5, 0.0, 5.0, 0.5, 0.0, 5.0, 0.5, 5.0, 0.0,
            5.5,
        ],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 1, 1, 1, 2, 2, 2];
    let fitted = NuSVC::<f64, LinearKernel>::new(LinearKernel)
        .with_nu(0.5)
        .with_tol(1e-6)
        .with_max_iter(200_000)
        .with_probability(true)
        .fit(&x, &y)
        .unwrap();
    let q = Array2::from_shape_vec((3, 2), vec![0.25, 0.25, 5.0, 0.25, 0.25, 5.0]).unwrap();
    let p = fitted.predict_proba(&q).unwrap();
    assert_eq!(p.dim(), (3, 3), "predict_proba shape must be (n, 3)");
    for s in 0..3 {
        let row_sum: f64 = (0..3).map(|c| p[[s, c]]).sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-9,
            "row {s} sums to {row_sum}, must be 1"
        );
        for c in 0..3 {
            assert!(
                p[[s, c]] >= 0.0 && p[[s, c]] <= 1.0,
                "p[{s},{c}] = {} out of [0,1]",
                p[[s, c]]
            );
        }
    }
    // probA_/probB_ length = n_pairs = 3 (structural contract).
    assert_eq!(fitted.prob_a().len(), 3, "probA_ length must be n_pairs=3");
    assert_eq!(fitted.prob_b().len(), 3, "probB_ length must be n_pairs=3");
    assert!(fitted.probability(), "probability() must be true");
}
