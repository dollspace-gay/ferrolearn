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
//! PIN 1-4 use the original public API. PIN 5-7 exercise the NEW libsvm-layout
//! fitted-attribute accessors (`FittedSVC::support`/`support_vectors`/
//! `n_support`/`dual_coef`/`intercept`/`coef`, `FittedSVR::support`/
//! `support_vectors`/`n_support`/`dual_coef`/`intercept`) added by the builder
//! (#636/#639/#640), pinning REQ-2/REQ-3/REQ-6/REQ-7 against the live oracle.
//! PIN 8-10 exercise the NEW REQ-4 `decision_function_shape` ovr/ovo support +
//! the binary 1-D `SvmScores::Binary` enum-variant contract (#637).

use ferrolearn_core::{Fit, Predict};
use ferrolearn_linear::svm::{LinearKernel, RbfKernel, SVC, SVR, SvmDecisionShape};
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

/// 3-class 9x2 separable linear set shared by PIN 8 / PIN 9.
fn three_class_9x2() -> (Array2<f64>, Array1<usize>) {
    let x = Array2::from_shape_vec(
        (9, 2),
        vec![
            0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 5.0, 0.0, 5.5, 0.0, 5.0, 0.5, 0.0, 5.0, 0.5, 5.0, 0.0,
            5.5,
        ],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 1, 1, 1, 2, 2, 2];
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

    // sklearn binary decision_function is 1-D shape (6,) (`-dec.ravel()`,
    // `_base.py:538-539`). ferrolearn now returns `SvmScores::Binary` of the
    // same 1-D shape; read it via `as_binary()`.
    let bin = df.as_binary().expect("binary decision_function is 1-D");
    assert_eq!(bin.len(), 6, "expected 6 values");

    // Live oracle values from SVC(kernel='linear',C=1.0).decision_function(X).
    let oracle: [f64; 6] = [
        -1.285333, -0.999733, -0.999733, 0.999467, 1.285067, 1.285067,
    ];
    for (i, &exp) in oracle.iter().enumerate() {
        let got = bin[i];
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
    // Binary: `SvmScores::Binary` 1-D (mechanical adaptation to the new enum
    // return type; this pin's REQ-1/gamma logic + oracle are unchanged).
    let bin = df.as_binary().expect("binary decision_function is 1-D");

    // Live oracle SVC(C=1.0) (default rbf, gamma='scale'=0.1184) values.
    let oracle: [f64; 6] = [
        -1.013804, -1.000632, -1.000319, 1.000308, 1.000335, 1.000308,
    ];
    for (i, &exp) in oracle.iter().enumerate() {
        let got = bin[i];
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

/// PIN 5 — REQ-2/REQ-3: binary fitted attributes + C-SVC fit correctness
/// (#635/#636).
///
/// Pins the NEW libsvm-layout accessors on `FittedSVC`
/// (`support`/`n_support`/`support_vectors`/`dual_coef`/`intercept`/`coef`,
/// `ferrolearn-linear/src/svm.rs`) against the live oracle. Simultaneously
/// verifies #635 — that `smo_binary` converged to libsvm's alpha (the
/// `dual_coef`/`intercept` values can only match if the SMO optimum matches).
///
/// Binary sign-flip contract: `sklearn/svm/_base.py:260-262`
/// (`if self._impl in ["c_svc","nu_svc"] and len(self.classes_)==2:
/// self.intercept_ *= -1; self.dual_coef_ = -self.dual_coef_`); linear-only
/// `coef_` at `sklearn/svm/_base.py:650-651` (raises `AttributeError` for
/// non-linear).
///
/// Live oracle, re-derived this session (R-CHAR-3):
/// ```text
/// python3 -c "import numpy as np; from sklearn.svm import SVC; \
///   X=np.array([[1.,1.],[2.,1.],[1.,2.],[5.,5.],[6.,5.],[5.,6.]]); y=np.array([0,0,0,1,1,1]); \
///   m=SVC(kernel='linear',C=1.0).fit(X,y); \
///   print(m.support_.tolist(), m.n_support_.tolist(), m.support_vectors_.tolist()); \
///   print(m.dual_coef_.tolist(), m.intercept_.tolist(), m.coef_.tolist())"
/// # support_ [1, 2, 3]  n_support_ [2, 1]
/// # support_vectors_ [[2.,1.],[1.,2.],[5.,5.]]
/// # dual_coef_ [[-0.0408, -0.0408, 0.0816]] (1,3)  intercept_ [-1.8565333]
/// # coef_ [[0.2856, 0.2856]] (1,2)
/// ```
///
/// Tracking: #635 / #636
#[test]
fn divergence_pin5_binary_fitted_attributes() {
    let (x, y) = binary_6x2();
    let model = SVC::<f64, LinearKernel>::new(LinearKernel)
        .with_c(1.0)
        .with_tol(1e-6)
        .with_max_iter(1_000_000);
    let fitted = model.fit(&x, &y).unwrap();

    // support_ == [1,2,3]
    let support = fitted.support();
    assert_eq!(
        support.to_vec(),
        vec![1usize, 2, 3],
        "support_: ferrolearn={:?}, sklearn=[1,2,3]",
        support.to_vec()
    );

    // n_support_ == [2,1]
    let n_support = fitted.n_support();
    assert_eq!(
        n_support,
        vec![2usize, 1],
        "n_support_: ferrolearn={n_support:?}, sklearn=[2,1]"
    );

    // support_vectors_ == [[2,1],[1,2],[5,5]] (rows in support_ order)
    let svs = fitted.support_vectors();
    assert_eq!(svs.dim(), (3, 2), "support_vectors_ shape");
    let oracle_svs: [[f64; 2]; 3] = [[2.0, 1.0], [1.0, 2.0], [5.0, 5.0]];
    for (r, row) in oracle_svs.iter().enumerate() {
        for (c, &exp) in row.iter().enumerate() {
            assert!(
                (svs[[r, c]] - exp).abs() < 1e-2,
                "support_vectors_[{r}][{c}]: ferrolearn={}, sklearn={exp}",
                svs[[r, c]]
            );
        }
    }

    // dual_coef_ shape (1,3) == [[-0.0408,-0.0408,0.0816]] (verifies #635 SMO).
    let dual = fitted.dual_coef();
    assert_eq!(dual.dim(), (1, 3), "dual_coef_ shape");
    let oracle_dual: [f64; 3] = [-0.0408, -0.0408, 0.0816];
    for (c, &exp) in oracle_dual.iter().enumerate() {
        assert!(
            (dual[[0, c]] - exp).abs() < 1e-2,
            "dual_coef_[0][{c}]: ferrolearn={}, sklearn={exp}, gap={}",
            dual[[0, c]],
            (dual[[0, c]] - exp).abs()
        );
    }

    // intercept_ == [-1.8565333]
    let intercept = fitted.intercept();
    assert_eq!(intercept.len(), 1, "intercept_ length");
    assert!(
        (intercept[0] - (-1.8565333)).abs() < 1e-2,
        "intercept_: ferrolearn={}, sklearn=-1.8565333, gap={}",
        intercept[0],
        (intercept[0] - (-1.8565333)).abs()
    );

    // coef_ == Some([[0.2856,0.2856]]) for linear kernel.
    let coef = fitted.coef();
    let coef = coef.expect("coef() must be Some for the linear kernel");
    assert_eq!(coef.dim(), (1, 2), "coef_ shape");
    let oracle_coef: [f64; 2] = [0.2856, 0.2856];
    for (c, &exp) in oracle_coef.iter().enumerate() {
        assert!(
            (coef[[0, c]] - exp).abs() < 1e-2,
            "coef_[0][{c}]: ferrolearn={}, sklearn={exp}, gap={}",
            coef[[0, c]],
            (coef[[0, c]] - exp).abs()
        );
    }

    // coef_ is None for a non-linear (RBF) kernel: sklearn raises
    // AttributeError (`sklearn/svm/_base.py:650-651`).
    let rbf_model = SVC::new(RbfKernel::<f64>::with_gamma(0.5))
        .with_c(1.0)
        .with_tol(1e-6)
        .with_max_iter(1_000_000);
    let rbf_fitted = rbf_model.fit(&x, &y).unwrap();
    assert!(
        rbf_fitted.coef().is_none(),
        "coef() must be None for a non-linear kernel (sklearn raises AttributeError)"
    );
}

/// PIN 6 — REQ-7: multiclass `dual_coef_` packing + per-pair `intercept_`
/// (#640).
///
/// Pins the libsvm `(n_class-1, n_SV)` row-packing of `dual_coef_` and the
/// per-ovo-pair `intercept_` (length `n_class*(n_class-1)/2`) for a 3-class
/// linear fit. Multiclass dual_coef has NO binary sign flip
/// (`sklearn/svm/_base.py:260` restricts the flip to `len(classes_)==2`); the
/// columns are the SVs in `support_` (per-class-grouped) order.
///
/// Live oracle, re-derived this session (R-CHAR-3):
/// ```text
/// python3 -c "import numpy as np; from sklearn.svm import SVC; \
///   X=np.array([[0.,0.],[0.5,0.],[0.,0.5],[5.,0.],[5.5,0.],[5.,0.5],[0.,5.],[0.5,5.],[0.,5.5]]); \
///   y=np.array([0,0,0,1,1,1,2,2,2]); m=SVC(kernel='linear',C=1.0).fit(X,y); \
///   print(m.support_.tolist(), m.n_support_.tolist()); \
///   print(np.round(m.dual_coef_,4).tolist(), np.round(m.intercept_,4).tolist())"
/// # support_ [1,2,3,5,6,7]  n_support_ [2,2,2]
/// # dual_coef_ [[0.0988,0,-0.0988,-0,-0.0988,-0],[0,0.0988,0,0.0494,-0,-0.0494]] (2,6)
/// # intercept_ [1.2222, 1.2222, 0.0]
/// ```
///
/// Tracking: #640
#[test]
fn divergence_pin6_multiclass_dual_coef_packing() {
    let x = Array2::from_shape_vec(
        (9, 2),
        vec![
            0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 5.0, 0.0, 5.5, 0.0, 5.0, 0.5, 0.0, 5.0, 0.5, 5.0, 0.0,
            5.5,
        ],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 1, 1, 1, 2, 2, 2];
    let model = SVC::<f64, LinearKernel>::new(LinearKernel)
        .with_c(1.0)
        .with_tol(1e-6)
        .with_max_iter(1_000_000);
    let fitted = model.fit(&x, &y).unwrap();

    // support_ == [1,2,3,5,6,7]
    let support = fitted.support();
    assert_eq!(
        support.to_vec(),
        vec![1usize, 2, 3, 5, 6, 7],
        "support_: ferrolearn={:?}, sklearn=[1,2,3,5,6,7]",
        support.to_vec()
    );

    // n_support_ == [2,2,2]
    let n_support = fitted.n_support();
    assert_eq!(
        n_support,
        vec![2usize, 2, 2],
        "n_support_: ferrolearn={n_support:?}, sklearn=[2,2,2]"
    );

    // dual_coef_ shape (2,6), libsvm packing.
    let dual = fitted.dual_coef();
    assert_eq!(dual.dim(), (2, 6), "dual_coef_ shape");
    let oracle_dual: [[f64; 6]; 2] = [
        [0.0988, 0.0, -0.0988, 0.0, -0.0988, 0.0],
        [0.0, 0.0988, 0.0, 0.0494, 0.0, -0.0494],
    ];
    for (r, row) in oracle_dual.iter().enumerate() {
        for (c, &exp) in row.iter().enumerate() {
            assert!(
                (dual[[r, c]] - exp).abs() < 1e-2,
                "dual_coef_[{r}][{c}]: ferrolearn={}, sklearn={exp}, gap={}",
                dual[[r, c]],
                (dual[[r, c]] - exp).abs()
            );
        }
    }

    // intercept_ length 3 == [1.2222, 1.2222, 0.0]
    let intercept = fitted.intercept();
    assert_eq!(intercept.len(), 3, "intercept_ length");
    let oracle_intercept: [f64; 3] = [1.2222, 1.2222, 0.0];
    for (i, &exp) in oracle_intercept.iter().enumerate() {
        assert!(
            (intercept[i] - exp).abs() < 1e-2,
            "intercept_[{i}]: ferrolearn={}, sklearn={exp}, gap={}",
            intercept[i],
            (intercept[i] - exp).abs()
        );
    }
}

/// PIN 7 — REQ-6: SVR fitted attributes (#639).
///
/// Pins the NEW libsvm-layout accessors on `FittedSVR`
/// (`support`/`dual_coef`/`intercept`, `ferrolearn-linear/src/svm.rs`) against
/// the live oracle. SVR has no binary sign flip (`sklearn/svm/_base.py:260`
/// restricts the flip to `c_svc`/`nu_svc`); `n_support_` has size 1
/// (`sklearn/svm/_base.py:680-682`); `dual_coef_` is `(1, n_SV)`.
///
/// Live oracle, re-derived this session (R-CHAR-3):
/// ```text
/// python3 -c "import numpy as np; from sklearn.svm import SVR; \
///   X=np.array([[1.],[2.],[3.],[4.],[5.],[6.]]); y=np.array([2.,4.,6.,8.,10.,12.]); \
///   m=SVR(kernel='linear',C=100.0,epsilon=0.1).fit(X,y); \
///   print(m.support_.tolist(), np.round(m.dual_coef_,4).tolist(), np.round(m.intercept_,4).tolist())"
/// # support_ [0, 5]  dual_coef_ [[-0.392, 0.392]] (1,2)  intercept_ [0.14]
/// ```
///
/// Tracking: #639
#[test]
fn divergence_pin7_svr_fitted_attributes() {
    let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let y = array![2.0f64, 4.0, 6.0, 8.0, 10.0, 12.0];
    let model = SVR::<f64, LinearKernel>::new(LinearKernel)
        .with_c(100.0)
        .with_epsilon(0.1)
        .with_tol(1e-6)
        .with_max_iter(1_000_000);
    let fitted = model.fit(&x, &y).unwrap();

    // support_ == [0,5]
    let support = fitted.support();
    assert_eq!(
        support.to_vec(),
        vec![0usize, 5],
        "support_: ferrolearn={:?}, sklearn=[0,5]",
        support.to_vec()
    );

    // dual_coef_ shape (1,2) == [[-0.392, 0.392]]
    let dual = fitted.dual_coef();
    assert_eq!(dual.dim(), (1, 2), "dual_coef_ shape");
    let oracle_dual: [f64; 2] = [-0.392, 0.392];
    for (c, &exp) in oracle_dual.iter().enumerate() {
        assert!(
            (dual[[0, c]] - exp).abs() < 1e-2,
            "dual_coef_[0][{c}]: ferrolearn={}, sklearn={exp}, gap={}",
            dual[[0, c]],
            (dual[[0, c]] - exp).abs()
        );
    }

    // intercept_ == [0.14]
    let intercept = fitted.intercept();
    assert_eq!(intercept.len(), 1, "intercept_ length");
    assert!(
        (intercept[0] - 0.14).abs() < 1e-2,
        "intercept_: ferrolearn={}, sklearn=0.14, gap={}",
        intercept[0],
        (intercept[0] - 0.14).abs()
    );
}

/// PIN 8 — REQ-4: multiclass OVR `decision_function` shape + values (#637).
///
/// Pins the NEW `SvmDecisionShape::Ovr` (default) path: `decision_function`
/// for `n_classes > 2` must return [`SvmScores::Multiclass`] shape
/// `(n_samples, n_classes)` = `(9, 3)`, computed via `_ovr_decision_function`
/// (`sklearn/utils/multiclass.py:520-562`) fed `dec<0`/`-dec`
/// (`sklearn/svm/_base.py:780`). It must NOT be the binary 1-D variant
/// (`as_binary()` is `None`).
///
/// Live oracle, re-derived this session (R-CHAR-3):
/// ```text
/// python3 -c "import numpy as np; from sklearn.svm import SVC; \
///   X=np.array([[0.,0.],[0.5,0.],[0.,0.5],[5.,0.],[5.5,0.],[5.,0.5],[0.,5.],[0.5,5.],[0.,5.5]]); \
///   y=np.array([0,0,0,1,1,1,2,2,2]); m=SVC(kernel='linear',C=1.0).fit(X,y); \
///   print(m.decision_function(X).shape, np.round(m.decision_function(X),4).tolist())"
/// # (9, 3)
/// # [[2.2366,0.8167,-0.1833],[2.2299,0.8431,-0.1905],[2.2299,-0.1905,0.8431],
/// #  [1.0606,2.2262,-0.2333],[1.0,2.2366,-0.2366],[1.0,2.2222,-0.2222],
/// #  [1.0606,-0.2333,2.2262],[1.0,-0.2222,2.2222],[1.0,-0.2366,2.2366]]
/// ```
///
/// Tracking: #637
#[test]
fn divergence_pin8_multiclass_ovr_decision_function() {
    let (x, y) = three_class_9x2();
    // Default decision_function_shape is Ovr (`SVC::new`).
    let model = SVC::<f64, LinearKernel>::new(LinearKernel)
        .with_c(1.0)
        .with_tol(1e-6)
        .with_max_iter(1_000_000);
    let fitted = model.fit(&x, &y).unwrap();
    let df = fitted.decision_function(&x).unwrap();

    // Multiclass must be the Multiclass variant, NOT Binary.
    assert!(
        df.as_binary().is_none(),
        "multiclass decision_function must NOT be the 1-D Binary variant"
    );
    let m = df
        .as_multiclass()
        .expect("multiclass (ovr) decision_function is 2-D");
    assert_eq!(m.dim(), (9, 3), "ovr decision_function shape");

    // Live oracle full (9,3) ovr matrix.
    let oracle: [[f64; 3]; 9] = [
        [2.2366, 0.8167, -0.1833],
        [2.2299, 0.8431, -0.1905],
        [2.2299, -0.1905, 0.8431],
        [1.0606, 2.2262, -0.2333],
        [1.0, 2.2366, -0.2366],
        [1.0, 2.2222, -0.2222],
        [1.0606, -0.2333, 2.2262],
        [1.0, -0.2222, 2.2222],
        [1.0, -0.2366, 2.2366],
    ];
    for (r, row) in oracle.iter().enumerate() {
        for (c, &exp) in row.iter().enumerate() {
            assert!(
                (m[[r, c]] - exp).abs() < 1e-2,
                "ovr decision_function[{r}][{c}]: ferrolearn={}, sklearn={exp}, gap={}",
                m[[r, c]],
                (m[[r, c]] - exp).abs()
            );
        }
    }
}

/// PIN 9 — REQ-4: multiclass OVO `decision_function` shape + values (#637).
///
/// Pins the NEW `SvmDecisionShape::Ovo` path: with
/// `with_decision_function_shape(SvmDecisionShape::Ovo)`, `decision_function`
/// must return the RAW one-vs-one decision values shape
/// `(n_samples, n_class*(n_class-1)/2)` = `(9, 3)`, in libsvm sign convention
/// (lower-index class is the `+1` side, `sklearn/svm/_base.py:520-524`). This
/// pins the raw-ovo column order + sign that feeds the ovr transform.
///
/// Live oracle, re-derived this session (R-CHAR-3):
/// ```text
/// python3 -c "import numpy as np; from sklearn.svm import SVC; \
///   X=np.array([[0.,0.],[0.5,0.],[0.,0.5],[5.,0.],[5.5,0.],[5.,0.5],[0.,5.],[0.5,5.],[0.,5.5]]); \
///   y=np.array([0,0,0,1,1,1,2,2,2]); \
///   m=SVC(kernel='linear',C=1.0,decision_function_shape='ovo').fit(X,y); \
///   print(m.decision_function(X).shape, np.round(m.decision_function(X)[0],4).tolist())"
/// # (9, 3)
/// # row0 [1.2222, 1.2222, 0.0]
/// ```
///
/// Tracking: #637
#[test]
fn divergence_pin9_multiclass_ovo_decision_function() {
    let (x, y) = three_class_9x2();
    let model = SVC::<f64, LinearKernel>::new(LinearKernel)
        .with_c(1.0)
        .with_tol(1e-6)
        .with_max_iter(1_000_000)
        .with_decision_function_shape(SvmDecisionShape::Ovo);
    let fitted = model.fit(&x, &y).unwrap();
    let df = fitted.decision_function(&x).unwrap();

    assert!(
        df.as_binary().is_none(),
        "multiclass ovo decision_function must NOT be the 1-D Binary variant"
    );
    let m = df
        .as_multiclass()
        .expect("multiclass (ovo) decision_function is 2-D");
    assert_eq!(m.dim(), (9, 3), "ovo decision_function shape (n_models=3)");

    // Live oracle row0 of the raw ovo matrix.
    let oracle_row0: [f64; 3] = [1.2222, 1.2222, 0.0];
    for (c, &exp) in oracle_row0.iter().enumerate() {
        assert!(
            (m[[0, c]] - exp).abs() < 1e-2,
            "ovo decision_function[0][{c}]: ferrolearn={}, sklearn={exp}, gap={}",
            m[[0, c]],
            (m[[0, c]] - exp).abs()
        );
    }
}

/// PIN 10 — REQ-4: binary 1-D `SvmScores::Binary` enum-variant contract (#637).
///
/// Focuses the SHAPE/enum-variant half of the binary decision_function
/// contract that PIN 1 covers value-wise: for `n_classes == 2`,
/// `decision_function` must return [`SvmScores::Binary`] (`as_binary()` is
/// `Some`, length 6, 1-D; `_base.py:538-539` `-dec.ravel()`) and NOT the
/// multiclass variant (`as_multiclass()` is `None`). Values are re-derived live
/// and asserted too (positive -> class 1).
///
/// Live oracle, re-derived this session (R-CHAR-3):
/// ```text
/// python3 -c "import numpy as np; from sklearn.svm import SVC; \
///   X=np.array([[1.,1.],[2.,1.],[1.,2.],[5.,5.],[6.,5.],[5.,6.]]); y=np.array([0,0,0,1,1,1]); \
///   m=SVC(kernel='linear',C=1.0).fit(X,y); \
///   print(m.decision_function(X).shape, np.round(m.decision_function(X),4).tolist())"
/// # (6,)
/// # [-1.2853, -0.9997, -0.9997, 0.9995, 1.2851, 1.2851]
/// ```
///
/// Tracking: #637
#[test]
fn divergence_pin10_binary_shape_contract() {
    let (x, y) = binary_6x2();
    let model = SVC::<f64, LinearKernel>::new(LinearKernel)
        .with_c(1.0)
        .with_tol(1e-6)
        .with_max_iter(1_000_000);
    let fitted = model.fit(&x, &y).unwrap();
    let df = fitted.decision_function(&x).unwrap();

    // Enum-variant contract: binary -> Binary 1-D, multiclass variant absent.
    assert!(
        df.as_multiclass().is_none(),
        "binary decision_function must NOT be the 2-D Multiclass variant"
    );
    let bin = df
        .as_binary()
        .expect("binary decision_function must be the 1-D Binary variant");
    assert_eq!(bin.len(), 6, "binary decision_function length (1-D (6,))");

    // Live oracle values (positive -> classes_[1]).
    let oracle: [f64; 6] = [-1.2853, -0.9997, -0.9997, 0.9995, 1.2851, 1.2851];
    for (i, &exp) in oracle.iter().enumerate() {
        assert!(
            (bin[i] - exp).abs() < 1e-2,
            "binary decision_function[{i}]: ferrolearn={}, sklearn={exp}, gap={}",
            bin[i],
            (bin[i] - exp).abs()
        );
    }
}

/// 4-class 12x2 set with a robustly-reproducible ovo VOTE TIE, shared by PIN 11.
///
/// Three tight 3-point clusters at the vertices of a (scalene) triangle —
/// class 1 lower-left `(-5,-3)`, class 2 lower-right `(5,-3)`, class 3 near
/// `(0,-1)` — plus class 0 up at `(0,5)`. A query well BELOW the triangle is
/// closer to all three of classes 1/2/3 than to class 0 (so class 0 loses every
/// pair it is in) while the three pairwise boundaries among 1/2/3 split their
/// votes 1-1-1 — producing the 3-way top tie at 2 votes each. Coordinates and
/// the tie query were located by a live-sklearn grid search (see PIN 11 doc).
fn four_class_tie_12x2() -> (Array2<f64>, Array1<usize>) {
    let x = Array2::from_shape_vec(
        (12, 2),
        vec![
            0.0, 5.0, 0.4, 5.0, 0.0, 5.4, // class 0 (up high)
            -5.0, -3.0, -5.4, -3.0, -5.0, -3.4, // class 1 (lower-left)
            5.0, -3.0, 5.4, -3.0, 5.0, -3.4, // class 2 (lower-right)
            0.0, -1.0, 0.4, -1.0, 0.0, -1.4, // class 3 (near center-bottom)
        ],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3];
    (x, y)
}

/// PIN 11 — REQ-5: ovo vote TIE-BREAK divergence (#638).
///
/// Divergence: `FittedSVC::predict` in `ferrolearn-linear/src/svm.rs` aggregates
/// the one-vs-one votes with `max_by_key(|&(_, &v)| v)`, which returns the LAST
/// maximum — so on a top-vote tie ferrolearn picks the HIGHER class index.
/// libsvm/sklearn's `SVC.predict` (`super().predict` -> libsvm `svm_predict`)
/// breaks ovo vote ties toward the LOWER class index
/// (`sklearn/svm/_base.py:813` `y = super().predict(X)` -> libsvm voting;
/// `_base.py:814` `return self.classes_.take(...)`).
///
/// ## Constructing the observable tie (live grid search, R-CHAR-3)
///
/// 3-class linear vote ties are ~measure-zero (3 bisectors meet at one point),
/// so this uses a 4-class config (6 ovo classifiers). A grid search over query
/// points with the live oracle found a robust query `q = (-0.21, -8.976)` below
/// the triangle of classes 1/2/3 where the vote count is `(0, 2, 2, 2)` — a
/// 3-way tie among classes 1, 2, 3.
///
/// ```text
/// python3 -c "
/// import numpy as np; from sklearn.svm import SVC; from itertools import combinations
/// X=np.array([[0,5],[0.4,5],[0,5.4],[-5,-3],[-5.4,-3],[-5,-3.4],
///             [5,-3],[5.4,-3],[5,-3.4],[0,-1],[0.4,-1],[0,-1.4]],dtype=float)
/// y=np.array([0,0,0,1,1,1,2,2,2,3,3,3])
/// mo=SVC(kernel='linear',C=1.0).fit(X,y)
/// m =SVC(kernel='linear',C=1.0,decision_function_shape='ovo').fit(X,y)
/// q=np.array([[-0.21,-8.976]])
/// dec=m.decision_function(q)[0]   # ovo, pair order (0,1)(0,2)(0,3)(1,2)(1,3)(2,3)
/// # dec = [-1.5361, -1.5599, -3.6587, 0.0420, -0.0442, 0.0450]
/// print(mo.predict(q))            # -> [1]   (libsvm LOWER-index tie-break)
/// print(mo.predict(X))            # -> [0,0,0,1,1,1,2,2,2,3,3,3] (non-tie)
/// "
/// ```
///
/// Per-pair winner derivation (libsvm: `dec>0` -> LOWER index of the pair wins;
/// this is exactly the sign convention ferrolearn reproduces — PIN 9 confirms
/// ferrolearn's raw ovo decisions match the oracle to 1e-2, and at this q
/// ferrolearn's own ovo dec `[-1.5361,-1.5599,-3.6587,0.0420,-0.0442,0.0450]`
/// matches the oracle to 4 decimals):
///   pair(0,1) dec=-1.5361 -> class 1 ; pair(0,2) dec=-1.5599 -> class 2
///   pair(0,3) dec=-3.6587 -> class 3 ; pair(1,2) dec=+0.0420 -> class 1
///   pair(1,3) dec=-0.0442 -> class 3 ; pair(2,3) dec=+0.0450 -> class 2
/// votes -> class0:0, class1:2, class2:2, class3:2  (3-way tie at 2)
///   - libsvm  lower-index rule -> class 1  (the oracle's answer)
///   - max_by_key last-max rule -> class 3  (ferrolearn's answer)
///
/// The nearest decision boundary is `min|dec| = 0.042`, well above PIN 9's
/// proven 1e-2 ovo agreement, so the tie is robustly reproduced by ferrolearn's
/// fit — the ONLY difference is the tie-break rule.
///
/// This pin FAILS today: ferrolearn `predict([-0.21,-8.976])` returns 3, the
/// oracle returns 1. It also asserts the non-tie training-point predictions
/// still match the oracle (so the pin is not solely about the tie).
///
/// Tracking: #638
#[test]
fn divergence_pin11_ovo_vote_tie_break_lower_index() {
    let (x, y) = four_class_tie_12x2();
    let model = SVC::<f64, LinearKernel>::new(LinearKernel)
        .with_c(1.0)
        .with_tol(1e-6)
        .with_max_iter(1_000_000);
    let fitted = model.fit(&x, &y).unwrap();

    // Non-tie sanity: the 12 training points predict their own class (the pin
    // is not only about the tie). Live oracle predict(X) == y.
    let train_pred = fitted.predict(&x).unwrap();
    let oracle_train = [0usize, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3];
    for (i, &exp) in oracle_train.iter().enumerate() {
        assert_eq!(
            train_pred[i], exp,
            "non-tie train predict[{i}]: ferrolearn={}, sklearn={exp}",
            train_pred[i]
        );
    }

    // THE TIE: q=(-0.21,-8.976) -> votes (0,2,2,2). libsvm picks the LOWER index
    // (class 1); ferrolearn's `max_by_key` last-max picks the HIGHER (class 3).
    let q = Array2::from_shape_vec((1, 2), vec![-0.21, -8.976]).unwrap();
    let tie_pred = fitted.predict(&q).unwrap();
    // Live oracle SVC(kernel='linear',C=1.0).predict([[-0.21,-8.976]]) == [1].
    assert_eq!(
        tie_pred[0], 1usize,
        "ovo vote-tie predict: ferrolearn={} (last-max -> higher index), \
         sklearn=1 (libsvm lower-index tie-break, _base.py:814)",
        tie_pred[0]
    );
}
