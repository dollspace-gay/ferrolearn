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

// ===========================================================================
// PIN 4 / PIN 5 — fitted-attribute accessors landed (REQ-1/#646, REQ-3/#648,
// REQ-4 score_samples/#649). These pins exercise the new public accessors
// (`support`/`support_vectors`/`n_support`/`dual_coef`/`intercept`/`offset`/
// `coef`/`score_samples`) against the live sklearn 1.5.2 oracle.
//
// The NOTE above (lines 13-19) is now superseded for these REQs — the
// accessors exist, so the pins below compile and assert real sklearn values.
// ===========================================================================

/// PIN 4 — #648/#649: hyperplane fitted attributes + `score_samples`.
///
/// Pins `FittedOneClassSVM::{intercept,offset,coef,score_samples,dual_coef}`
/// against the live `OneClassSVM(kernel='linear', nu=0.5)` oracle on the shared
/// 7x2 set. These are the HYPERPLANE-level attributes — they are IDENTICAL
/// across the optimal face (the α-decomposition non-uniqueness affects only the
/// `support_`/`dual_coef_` *vector*, NOT `coef_`/`intercept_`/`offset_`/the
/// decision function). Per the task brief this pin asserts only the
/// scale-invariant / hyperplane quantities (incl. `dual_coef_.sum() = ν·n`),
/// NOT the exact `support_`/`dual_coef_` vector (that is PIN 5's job on a
/// non-degenerate set).
///
/// Oracle (re-derived live, R-CHAR-3):
/// ```text
/// python3 -c "import numpy as np; from sklearn.svm import OneClassSVM; \
/// X=np.array([[0,0],[0.1,0.1],[-0.1,0.1],[0.1,-0.1],[0,0.2],[0.2,0],[3,3]],dtype=float); \
/// m=OneClassSVM(kernel='linear',nu=0.5).fit(X); \
/// print('intercept_',m.intercept_.tolist()); \
/// print('offset_',float(np.atleast_1d(m.offset_)[0])); \
/// print('coef_',m.coef_.ravel().tolist()); \
/// print('dual_sum',float(m.dual_coef_.sum())); \
/// print('score_samples',np.round(m.score_samples(X),6).tolist())"
/// # intercept_     [-0.01]
/// # offset_        0.01
/// # coef_          [0.05, 0.05]
/// # dual_sum       3.5            (= nu*n = 0.5*7)
/// # score_samples  [0.0, 0.01, 0.0, 0.0, 0.01, 0.01, 0.3]
/// ```
#[test]
fn divergence_pin4_hyperplane_attrs_648_649() {
    // LIVE oracle values (R-CHAR-3) — never copied from ferrolearn.
    let oracle_intercept: f64 = -0.01;
    let oracle_offset: f64 = 0.01;
    let oracle_coef: [f64; 2] = [0.05, 0.05];
    let oracle_dual_sum: f64 = 3.5; // nu*n
    let oracle_score_samples: [f64; 7] = [0.0, 0.01, 0.0, 0.0, 0.01, 0.01, 0.3];

    let x = fixture();
    let model = OneClassSVM::new(LinearKernel)
        .with_nu(0.5)
        .with_tol(1e-9)
        .with_max_iter(1_000_000);
    let fitted = model.fit(&x, &()).expect("fit should succeed");

    // intercept_ == [-0.01]
    let intercept = fitted.intercept();
    assert_eq!(intercept.len(), 1, "intercept_ length 1");
    eprintln!(
        "PIN4 intercept_ oracle={oracle_intercept} ferro={}",
        intercept[0]
    );
    assert!(
        (intercept[0] - oracle_intercept).abs() < 1e-2,
        "intercept_: ferrolearn={} expected (sklearn)={oracle_intercept}",
        intercept[0]
    );

    // offset_ == 0.01 (== -intercept_)
    let offset = fitted.offset();
    eprintln!("PIN4 offset_ oracle={oracle_offset} ferro={offset}");
    assert!(
        (offset - oracle_offset).abs() < 1e-2,
        "offset_: ferrolearn={offset} expected (sklearn)={oracle_offset}"
    );

    // coef_ == [[0.05, 0.05]] (linear kernel exposes it).
    let coef = fitted.coef().expect("linear kernel exposes coef_");
    assert_eq!(coef.dim(), (1, 2), "coef_ shape (1, n_features)");
    eprintln!(
        "PIN4 coef_ oracle={oracle_coef:?} ferro=[{}, {}]",
        coef[[0, 0]],
        coef[[0, 1]]
    );
    assert!(
        (coef[[0, 0]] - oracle_coef[0]).abs() < 1e-2,
        "coef_[0][0]: ferrolearn={} expected (sklearn)={}",
        coef[[0, 0]],
        oracle_coef[0]
    );
    assert!(
        (coef[[0, 1]] - oracle_coef[1]).abs() < 1e-2,
        "coef_[0][1]: ferrolearn={} expected (sklearn)={}",
        coef[[0, 1]],
        oracle_coef[1]
    );

    // dual_coef_.sum() == 3.5 (= nu*n; scale-invariant of the decomposition).
    let dual_sum: f64 = fitted.dual_coef().iter().sum();
    eprintln!("PIN4 dual_coef_.sum() oracle={oracle_dual_sum} ferro={dual_sum}");
    assert!(
        (dual_sum - oracle_dual_sum).abs() < 1e-2,
        "dual_coef_.sum(): ferrolearn={dual_sum} expected (sklearn)={oracle_dual_sum} (= nu*n)"
    );

    // score_samples == decision_function + offset_ == [0,0.01,0,0,0.01,0.01,0.3]
    let scores = fitted
        .score_samples(&x)
        .expect("score_samples should succeed");
    assert_eq!(scores.len(), oracle_score_samples.len());
    eprintln!(
        "PIN4 score_samples oracle={oracle_score_samples:?} ferro={:?}",
        scores.to_vec()
    );
    for (i, &exp) in oracle_score_samples.iter().enumerate() {
        assert!(
            (scores[i] - exp).abs() < 1e-2,
            "score_samples[{i}]: ferrolearn={} expected (sklearn)={exp}",
            scores[i]
        );
    }
}

/// PIN 4b — #648: `coef_` is `None` for a non-linear (rbf) kernel.
///
/// sklearn raises `AttributeError` when accessing `coef_` on a non-linear
/// `OneClassSVM` (`sklearn/svm/_base.py:650-666` gates `coef_` on
/// `kernel == "linear"`). ferrolearn's `coef()` returns `None` for non-linear.
#[test]
fn divergence_pin4b_coef_none_for_rbf_648() {
    let x = fixture();
    let model = OneClassSVM::new(RbfKernel::with_gamma(1.0))
        .with_nu(0.5)
        .with_tol(1e-9)
        .with_max_iter(1_000_000);
    let fitted = model.fit(&x, &()).expect("rbf fit should succeed");
    assert!(
        fitted.coef().is_none(),
        "coef_ must be None for rbf (sklearn raises AttributeError)"
    );
}

/// The non-degenerate PIN-5 fixture: an asymmetric 5-point cluster + a clear
/// outlier `[5,5]`. Unlike the symmetric 7x2 set, the optimal α-decomposition
/// here is UNIQUE (verified live: see the module-level note in the PIN-5 test).
fn fixture_nondegenerate() -> Array2<f64> {
    Array2::from_shape_vec(
        (6, 2),
        vec![0.0, 0.0, 1.0, 0.2, 0.3, 1.1, 1.4, 1.3, 0.7, 0.5, 5.0, 5.0],
    )
    .unwrap()
}

/// PIN 5 — #646: SV decomposition on a NON-DEGENERATE set (unique optimum).
///
/// The toy 7x2 set has a DEGENERATE optimal face (margin points 1,4,5 satisfy
/// `0.5·x₁ = 0.25·x₄ + 0.25·x₅`), so libsvm and ferrolearn's SMO can land on
/// different vertices (4 SVs `[0,1,2,3]` vs 5 SVs `[0,2,3,4,5]`) of the SAME
/// hyperplane — genuine α-decomposition non-uniqueness, NOT a solver bug.
///
/// This pin instead uses an asymmetric cluster + clear outlier where the
/// decomposition is UNIQUE. Uniqueness was verified LIVE by perturbing X by
/// N(0, 1e-4) noise 20× and confirming (a) `support_` is invariant `[0,1,2,4]`
/// and (b) `dual_coef_` varies *continuously* (std ≈ 2e-4 ∝ the perturbation,
/// no jump to another vertex). `coef_ = [1.398, 1.088]` is well away from 0 and
/// the two free margin SVs (rows 1,2) are in general position:
/// ```text
/// python3 -c "import numpy as np; from sklearn.svm import OneClassSVM; \
/// X0=np.array([[0,0],[1,0.2],[0.3,1.1],[1.4,1.3],[0.7,0.5],[5,5]],dtype=float); \
/// rng=np.random.default_rng(0); base=OneClassSVM(kernel='linear',nu=0.5).fit(X0); \
/// duals=[]; \
/// [ ( (lambda m: (duals.append(m.dual_coef_.ravel()), \
///      None if m.support_.tolist()==base.support_.tolist() else print('CHANGED'))) \
///    (OneClassSVM(kernel='linear',nu=0.5).fit(X0+rng.normal(scale=1e-4,size=X0.shape))) ) \
///   for _ in range(20) ]; \
/// print('std',np.round(np.array(duals).std(0),6).tolist())"
/// # (no 'CHANGED' printed; support stable)  std [0.0, 0.00026, 0.00026, 0.0]
/// ```
///
/// Oracle (re-derived live, R-CHAR-3):
/// ```text
/// python3 -c "import numpy as np; from sklearn.svm import OneClassSVM; \
/// X=np.array([[0,0],[1,0.2],[0.3,1.1],[1.4,1.3],[0.7,0.5],[5,5]],dtype=float); \
/// m=OneClassSVM(kernel='linear',nu=0.5).fit(X); \
/// print(m.support_.tolist(), m.n_support_.tolist(), m.dual_coef_.ravel().tolist(), \
///       m.intercept_.tolist())"
/// # support_   [0, 1, 2, 4]
/// # n_support_ [4]
/// # dual_coef_ [1.0, 0.5692307470253847, 0.4307692529746154, 1.0]  (sum 3.0 = nu*n)
/// # intercept_ [-1.6159999734144956]
/// ```
///
/// If this pin is GREEN, the toy-set divergence is degeneracy-ONLY: REQ-1's
/// `support_`/`dual_coef_` are CORRECT on non-degenerate inputs and #646 can
/// close as a documented non-uniqueness boundary. If it is RED, the solver has
/// a real working-set-selection bug and a fixer is required.
#[test]
fn divergence_pin5_sv_decomposition_nondegenerate_646() {
    // LIVE oracle (R-CHAR-3) — never copied from ferrolearn.
    let oracle_support: [usize; 4] = [0, 1, 2, 4];
    let oracle_n_support: usize = 4;
    let oracle_dual: [f64; 4] = [1.0, 0.5692307470253847, 0.4307692529746154, 1.0];
    let oracle_dual_sum: f64 = 3.0; // nu*n = 0.5*6
    let oracle_intercept: f64 = -1.6159999734144956;

    let x = fixture_nondegenerate();
    let model = OneClassSVM::new(LinearKernel)
        .with_nu(0.5)
        .with_tol(1e-9)
        .with_max_iter(1_000_000);
    let fitted = model.fit(&x, &()).expect("fit should succeed");

    // support_ == [0,1,2,4]
    let support = fitted.support();
    eprintln!(
        "PIN5 support_ oracle={oracle_support:?} ferro={:?}",
        support.to_vec()
    );
    assert_eq!(
        support.len(),
        oracle_support.len(),
        "support_ length: ferrolearn={:?} expected (sklearn)={oracle_support:?}",
        support.to_vec()
    );
    for (i, &exp) in oracle_support.iter().enumerate() {
        assert_eq!(
            support[i], exp,
            "support_[{i}]: ferrolearn={} expected (sklearn)={exp}",
            support[i]
        );
    }

    // n_support_ == [4]
    let n_support = fitted.n_support();
    eprintln!("PIN5 n_support_ oracle=[{oracle_n_support}] ferro={n_support:?}");
    assert_eq!(n_support.len(), 1, "n_support_ length 1 for one-class");
    assert_eq!(
        n_support[0], oracle_n_support,
        "n_support_[0]: ferrolearn={} expected (sklearn)={oracle_n_support}",
        n_support[0]
    );

    // dual_coef_ == [[1.0, 0.5692.., 0.4308.., 1.0]] (sum 3.0 = nu*n)
    let dual = fitted.dual_coef();
    assert_eq!(
        dual.dim(),
        (1, oracle_dual.len()),
        "dual_coef_ shape (1, n_SV)"
    );
    let dual_vec: Vec<f64> = dual.iter().copied().collect();
    eprintln!("PIN5 dual_coef_ oracle={oracle_dual:?} ferro={dual_vec:?}");
    for (i, &exp) in oracle_dual.iter().enumerate() {
        assert!(
            (dual_vec[i] - exp).abs() < 1e-2,
            "dual_coef_[{i}]: ferrolearn={} expected (sklearn)={exp}",
            dual_vec[i]
        );
    }
    let dual_sum: f64 = dual_vec.iter().sum();
    assert!(
        (dual_sum - oracle_dual_sum).abs() < 1e-2,
        "dual_coef_.sum(): ferrolearn={dual_sum} expected (sklearn)={oracle_dual_sum} (= nu*n)"
    );

    // intercept_ == [-1.616]
    let intercept = fitted.intercept();
    assert_eq!(intercept.len(), 1, "intercept_ length 1");
    eprintln!(
        "PIN5 intercept_ oracle={oracle_intercept} ferro={}",
        intercept[0]
    );
    assert!(
        (intercept[0] - oracle_intercept).abs() < 1e-2,
        "intercept_: ferrolearn={} expected (sklearn)={oracle_intercept}",
        intercept[0]
    );
}
