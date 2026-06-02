//! Divergence + characterization pins for `QuadraticDiscriminantAnalysis`
//! against the live scikit-learn 1.5.2 oracle
//! (`sklearn/discriminant_analysis.py`, commit 156ef14).
//!
//! QDA is DETERMINISTIC (no RNG, closed-form per-class Gaussian fit), so exact
//! value parity is testable. sklearn fits a per-class SVD of the centered data
//! (`scalings_ = S²/(n_k-1)`, `rotations_ = Vt.T`) and evaluates the Mahalanobis
//! term in the rotated/scaled frame (`discriminant_analysis.py:962-976`);
//! ferrolearn directly inverts the per-class covariance via Cholesky
//! (`qda.rs::cholesky_inv_and_logdet`). For FULL-RANK covariances the two are
//! mathematically identical (`Σ = V·D·Vᵀ ⇒ Σ⁻¹ = V·D⁻¹·Vᵀ`,
//! `log|Σ| = Σ log(D)`), and the `reg_param` blend commutes with the
//! eigendecomposition, so the per-class log-posterior must agree to machine
//! precision.
//!
//! Every expected value below is produced by RUNNING scikit-learn 1.5.2 (the
//! live oracle), never copied from ferrolearn (goal.md R-CHAR-3). The exact
//! python invocation that produced each constant is recorded in a comment.
//!
//! # Core characterization pins (R-CHAR-3 coverage the unit lacked)
//!
//! `qda_decision_function_multiclass`, `qda_predict_multiclass`,
//! `qda_predict_proba_multiclass`, `qda_reg_param` LOCK the
//! `decision_function`/`predict`/`predict_proba` core to the live oracle on a
//! fixed 3-class dataset. If the inversion path is correct they PASS (the
//! intended outcome — they are the missing characterization coverage); if it
//! diverges they FAIL and pin it (#575/#576/#577/#579).
//!
//! # Divergence pin
//!
//! `qda_rank_deficient_class` pins #583: a perfectly-collinear class gives a
//! singular covariance. sklearn's SVD path produces predictions (with a
//! "Variables are collinear" warning, `discriminant_analysis.py:945-947`);
//! ferrolearn's Cholesky inversion returns `FerroError::NumericalInstability`.

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Predict};
use ferrolearn_linear::QDA;
use ndarray::{Array1, Array2, array};

// ===========================================================================
// Fixed 3-class, 2-feature, 4-points-per-class dataset shared by the core
// characterization pins. Chosen with OVERLAP between classes so predict_proba
// is non-degenerate (rows are not exactly [1,0,0]) and the softmax is actually
// exercised. All classes are full-rank (the inversion ≡ SVD regime).
//
//   X = [[0,0],[1,0.5],[0.5,1],[1,1],
//        [2,2],[3,2.5],[2.5,3],[3,3],
//        [1,3],[2,4],[1.5,3.5],[2,3]]
//   y = [0,0,0,0, 1,1,1,1, 2,2,2,2]
// ===========================================================================

fn mc_x() -> Array2<f64> {
    Array2::from_shape_vec(
        (12, 2),
        vec![
            0.0, 0.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, // class 0
            2.0, 2.0, 3.0, 2.5, 2.5, 3.0, 3.0, 3.0, // class 1
            1.0, 3.0, 2.0, 4.0, 1.5, 3.5, 2.0, 3.0, // class 2
        ],
    )
    .unwrap()
}

fn mc_y() -> Array1<usize> {
    Array1::from(vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
}

/// REQ-1 (#575) characterization: the `(n, n_classes)` raw per-class
/// log-posterior matches the live `QuadraticDiscriminantAnalysis()
/// ._decision_function` oracle on full-rank multiclass data.
///
/// Oracle (live sklearn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; from sklearn.discriminant_analysis import \
///   QuadraticDiscriminantAnalysis as Q; \
///   X=np.array([[0.,0.],[1.,.5],[.5,1.],[1.,1.],[2.,2.],[3.,2.5],[2.5,3.],[3.,3.],\
///               [1.,3.],[2.,4.],[1.5,3.5],[2.,3.]]); \
///   y=np.array([0,0,0,0,1,1,1,1,2,2,2,2]); \
///   m=Q().fit(X,y); \
///   [print(r) for r in m._decision_function(X).tolist()]"
/// ```
#[test]
fn qda_decision_function_multiclass() {
    // Live sklearn 1.5.2 `_decision_function(X)` (the raw (n,3) log-posterior).
    const SK_DEC: [[f64; 3]; 12] = [
        [-0.4074110039349128, -17.74074433726824, -24.384585373494144],
        [-0.15741100393491303, -9.490744337268247, -18.82208537349415],
        [
            -0.15741100393491325,
            -9.490744337268243,
            -11.822085373494144,
        ],
        [0.2592556627317537, -6.407411003934912, -12.38458537349414],
        [-4.407411003934912, -0.4074110039349128, -6.384585373494139],
        [
            -12.157411003934909,
            -0.15741100393491303,
            -9.822085373494133,
        ],
        [
            -12.15741100393491,
            -0.15741100393491325,
            -2.8220853734941365,
        ],
        [-14.40741100393491, 0.2592556627317537, -6.384585373494136],
        [-16.40741100393491, -12.40741100393491, -0.3845853734941369],
        [
            -26.40741100393491,
            -11.740744337268248,
            -0.38458537349413746,
        ],
        [
            -20.740744337268247,
            -11.407411003934914,
            0.36541462650586287,
        ],
        [
            -11.740744337268247,
            -2.4074110039349135,
            -0.6345853734941371,
        ],
    ];

    let fitted = QDA::<f64>::new().fit(&mc_x(), &mc_y()).unwrap();
    let dec = fitted.decision_function(&mc_x()).unwrap();

    assert_eq!(dec.dim(), (12, 3), "multiclass decision_function shape");
    for i in 0..12 {
        for c in 0..3 {
            assert!(
                (dec[[i, c]] - SK_DEC[i][c]).abs() < 1e-6,
                "decision_function[{i}][{c}]: sklearn {}, ferrolearn {}",
                SK_DEC[i][c],
                dec[[i, c]]
            );
        }
    }
}

/// REQ-2 (#576) characterization: `predict` returns
/// `classes_.take(argmax over raw per-class scores)`, label-for-label matching
/// the live `QuadraticDiscriminantAnalysis().predict` oracle.
///
/// Oracle (live sklearn 1.5.2):
/// ```text
/// python3 -c "...same X,y as above...; m=Q().fit(X,y); print(m.predict(X).tolist())"
/// # -> [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
/// ```
#[test]
fn qda_predict_multiclass() {
    const SK_PRED: [usize; 12] = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];

    let fitted = QDA::<f64>::new().fit(&mc_x(), &mc_y()).unwrap();
    let pred = fitted.predict(&mc_x()).unwrap();

    assert_eq!(pred.len(), 12);
    for i in 0..12 {
        assert_eq!(
            pred[i], SK_PRED[i],
            "predict[{i}]: sklearn {}, ferrolearn {}",
            SK_PRED[i], pred[i]
        );
    }
}

/// REQ-3 (#577) characterization: `predict_proba` is the softmax over the raw
/// per-class scores, matching the live `predict_proba` oracle. The dataset has
/// class overlap so the probabilities are non-degenerate (rows not [1,0,0]).
///
/// Oracle (live sklearn 1.5.2):
/// ```text
/// python3 -c "...same X,y as above...; m=Q().fit(X,y); \
///   [print(r) for r in m.predict_proba(X).tolist()]"
/// ```
#[test]
fn qda_predict_proba_multiclass() {
    // Live sklearn 1.5.2 `predict_proba(X)`.
    const SK_PROBA: [[f64; 3]; 12] = [
        [
            0.999999970297428,
            2.9663949112254623e-08,
            3.862295221274717e-11,
        ],
        [
            0.9999115729964404,
            8.841916932624201e-05,
            7.834233466315203e-09,
        ],
        [
            0.9999029903831606,
            8.841841039159517e-05,
            8.59120644787497e-06,
        ],
        [
            0.9987257645698585,
            0.0012710121662605474,
            3.2232638808630397e-06,
        ],
        [0.0179415289306429, 0.9795742883792382, 0.002484182690118855],
        [
            6.143784553760392e-06,
            0.9999303735705671,
            6.348264487904938e-05,
        ],
        [
            5.744250716069891e-06,
            0.9349043271524173,
            0.0650899285968665,
        ],
        [
            4.2636576205868324e-07,
            0.9986992506971732,
            0.0013003229370646722,
        ],
        [
            1.0999490988768489e-07,
            6.005518592930001e-06,
            0.9999938844864972,
        ],
        [
            4.993733309887739e-12,
            1.1697087708651372e-05,
            0.9999883029072977,
        ],
        [
            6.818805120469953e-10,
            7.711226203458235e-06,
            0.9999922880919161,
        ],
        [
            1.2838663370908406e-05,
            0.14518942197941162,
            0.8547977393572175,
        ],
    ];

    let fitted = QDA::<f64>::new().fit(&mc_x(), &mc_y()).unwrap();
    let proba = fitted.predict_proba(&mc_x()).unwrap();

    assert_eq!(proba.dim(), (12, 3));
    for i in 0..12 {
        // Rows sum to 1 (softmax normalization).
        let row_sum: f64 = (0..3).map(|c| proba[[i, c]]).sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-12,
            "predict_proba row {i} must sum to 1, got {row_sum}"
        );
        for c in 0..3 {
            assert!(
                (proba[[i, c]] - SK_PROBA[i][c]).abs() < 1e-6,
                "predict_proba[{i}][{c}]: sklearn {}, ferrolearn {}",
                SK_PROBA[i][c],
                proba[[i, c]]
            );
        }
    }
}

/// REQ-5 (#579) characterization: `reg_param=0.5` pins the covariance-diagonal
/// blend `(1-reg)·Σ + reg·I` against sklearn's singular-value blend
/// `S2 = (1-reg)·S²/(n_k-1) + reg` (`discriminant_analysis.py:949`). The two
/// are algebraically equal; this confirms ferrolearn's regularized
/// `predict_proba` matches the live `QuadraticDiscriminantAnalysis(reg_param=0.5)`
/// oracle.
///
/// Oracle (live sklearn 1.5.2):
/// ```text
/// python3 -c "...same X,y as above...; m=Q(reg_param=0.5).fit(X,y); \
///   [print(r) for r in m.predict_proba(X).tolist()]"
/// ```
#[test]
fn qda_reg_param() {
    // Live sklearn 1.5.2 `Q(reg_param=0.5).predict_proba(X)`.
    const SK_PROBA: [[f64; 3]; 12] = [
        [
            0.999883432675969,
            7.832356607521482e-05,
            3.8243757955836264e-05,
        ],
        [
            0.9926817927143142,
            0.00610739951530393,
            0.0012108077703819023,
        ],
        [
            0.9882094946790643,
            0.006079884040502988,
            0.005710621280432689,
        ],
        [
            0.9642510185131026,
            0.025406067336095878,
            0.010342914150801663,
        ],
        [0.07930769416173672, 0.7028569118423956, 0.2178353939958676],
        [
            0.0012928253187999112,
            0.8999008031388952,
            0.09880637154230483,
        ],
        [
            0.0009441433993809931,
            0.6571927320930967,
            0.3418631245075223,
        ],
        [
            0.00027592370509133267,
            0.8225169725061537,
            0.17720710378875493,
        ],
        [0.01317904217681388, 0.11679801037394374, 0.8700229474492424],
        [
            5.429576995439454e-05,
            0.16185340910813265,
            0.8380922951219129,
        ],
        [0.0008442183816317572, 0.137217192754564, 0.8619385888638041],
        [0.002715419382021481, 0.4413576307496788, 0.5559269498682997],
    ];

    let fitted = QDA::<f64>::new()
        .with_reg_param(0.5)
        .fit(&mc_x(), &mc_y())
        .unwrap();
    let proba = fitted.predict_proba(&mc_x()).unwrap();

    for i in 0..12 {
        for c in 0..3 {
            assert!(
                (proba[[i, c]] - SK_PROBA[i][c]).abs() < 1e-6,
                "reg_param=0.5 predict_proba[{i}][{c}]: sklearn {}, ferrolearn {}",
                SK_PROBA[i][c],
                proba[[i, c]]
            );
        }
    }
}

// ===========================================================================
// #583 — rank-deficient / collinear class. DIVERGENCE PIN.
// ===========================================================================

/// Divergence (#583): a perfectly-collinear class has a singular covariance.
/// sklearn's per-class SVD path produces predictions (emitting a "Variables are
/// collinear" warning, NOT an error); ferrolearn's Cholesky inversion rejects
/// the class with `FerroError::NumericalInstability`.
///
/// sklearn site: `sklearn/discriminant_analysis.py:944-947`
///   `_, S, Vt = np.linalg.svd(Xgc, full_matrices=False)`
///   `rank = np.sum(S > self.tol)`
///   `if rank < n_features: warnings.warn("Variables are collinear")`
/// — the SVD still yields `scalings_`/`rotations_` (a zero scaling for the
/// collapsed axis), so `fit`/`predict` succeed.
///
/// ferrolearn site: `ferrolearn-linear/src/qda.rs::cholesky_inv_and_logdet`
///   `if s <= F::zero() { return Err(FerroError::NumericalInstability { ... }) }`
/// — a singular (collinear) class covariance is not positive definite, so the
/// Cholesky factorization rejects it and `fit` returns Err.
///
/// Dataset: class 0 has feature-1 constant (== 0) across all four samples, so
/// its centered covariance is rank 1 (singular); class 1 is a full-rank cluster.
///
/// Oracle (live sklearn 1.5.2):
/// ```text
/// python3 -W ignore -c "import numpy as np; \
///   from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as Q; \
///   X=np.array([[0.,0.],[1.,0.],[2.,0.],[3.,0.],[8.,8.],[9.,7.5],[7.5,9.],[8.5,8.5]]); \
///   y=np.array([0,0,0,0,1,1,1,1]); \
///   m=Q().fit(X,y); print(m.predict(X).tolist())"
/// # warns 'Variables are collinear'; -> [0, 0, 0, 0, 0, 0, 0, 0]
/// ```
/// sklearn's singular class-0 Gaussian (one scaling == 0 ⇒ infinite precision
/// along the collapsed axis) dominates, so it deterministically predicts class
/// 0 for every sample. ferrolearn returns Err(NumericalInstability) from fit.
///
/// Tracking: #583
#[test]
fn qda_rank_deficient_class() {
    // Live sklearn 1.5.2 predictions on the training X (see invocation above).
    const SK_PRED: [usize; 8] = [0, 0, 0, 0, 0, 0, 0, 0];

    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0,
            0.0, // class 0: feature 1 constant => collinear
            8.0, 8.0, 9.0, 7.5, 7.5, 9.0, 8.5, 8.5, // class 1: full rank
        ],
    )
    .unwrap();
    let y = Array1::from(vec![0, 0, 0, 0, 1, 1, 1, 1]);

    // sklearn: fit + predict succeed (warning only). ferrolearn currently
    // returns Err(NumericalInstability) from the Cholesky inversion, so this fit
    // FAILS and the divergence is pinned.
    let fitted = QDA::<f64>::new().fit(&x, &y).expect(
        "rank-deficient (#583): sklearn handles a collinear class via SVD and \
         predicts (with a 'Variables are collinear' warning); ferrolearn must \
         not return NumericalInstability",
    );
    let pred = fitted
        .predict(&x)
        .expect("predict after rank-deficient fit");

    for i in 0..8 {
        assert_eq!(
            pred[i], SK_PRED[i],
            "rank-deficient predict[{i}]: sklearn {}, ferrolearn {}",
            SK_PRED[i], pred[i]
        );
    }
}

// ===========================================================================
// REQ-6 (#580) — provided `priors` constructor argument.
// ===========================================================================

/// REQ-6 (#580): `QDA::with_priors([0.9, 0.1])` uses the provided priors
/// VERBATIM (no renormalization, no sign/sum check), matching sklearn
/// `QuadraticDiscriminantAnalysis(priors=[0.9,0.1])`
/// (`discriminant_analysis.py:341` `"priors": ["array-like", None]`; `:351,359`
/// the constructor arg; `:921-924` `self.priors_ = np.array(self.priors)`; the
/// `+ np.log(self.priors_)` term in `_decision_function`, `:976`). The provided
/// priors shift the per-class log-posterior by `log(0.9)`/`log(0.1)` versus the
/// empirical default, observably moving `predict_proba` (row 4 below: empirical
/// proba[0] ≈ 0.0180 vs provided ≈ 0.1415).
///
/// `predict_proba` is the cross-shape-stable `(n, 2)` assertion (sklearn's
/// binary `decision_function` is `(n,)` = col1−col0; ferrolearn's lib returns
/// `(n, 2)` — the binding ABI reshape is REQ-7/#581, out of scope here).
///
/// Oracle (live sklearn 1.5.2, the OVERLAPPING 2-class dataset below so the
/// proba is non-degenerate):
/// ```text
/// python3 -c "import numpy as np; \
///   from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as Q; \
///   X=np.array([[0.,0.],[1.,.5],[.5,1.],[1.,1.],[2.,2.],[3.,2.5],[2.5,3.],[3.,3.]]); \
///   y=np.array([0,0,0,0,1,1,1,1]); \
///   m=Q(priors=[0.9,0.1]).fit(X,y); \
///   print(repr(m.predict_proba(X).tolist())); print(repr(m.priors_.tolist()))"
/// ```
#[test]
fn qda_provided_priors() {
    // Live sklearn 1.5.2 `Q(priors=[0.9,0.1]).predict_proba(X)`.
    const SK_PROBA: [[f64; 2]; 8] = [
        [0.9999999967040056, 3.295994432841995e-09],
        [0.9999901748755718, 9.825124428190748e-06],
        [0.9999901748755718, 9.825124428190783e-06],
        [0.9998586162364907, 0.00014138376350926758],
        [0.1415135502417861, 0.8584864497582139],
        [5.529485349005703e-05, 0.99994470514651],
        [5.5294853490056925e-05, 0.99994470514651],
        [3.842274951061619e-06, 0.9999961577250489],
    ];

    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            0.0, 0.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, // class 0
            2.0, 2.0, 3.0, 2.5, 2.5, 3.0, 3.0, 3.0, // class 1
        ],
    )
    .unwrap();
    let y = Array1::from(vec![0, 0, 0, 0, 1, 1, 1, 1]);

    let fitted = QDA::<f64>::new()
        .with_priors(array![0.9, 0.1])
        .fit(&x, &y)
        .unwrap();

    // priors_ is the provided vector verbatim.
    assert_eq!(fitted.priors().to_vec(), vec![0.9, 0.1], "priors_ verbatim");

    let proba = fitted.predict_proba(&x).unwrap();
    assert_eq!(proba.dim(), (8, 2));
    for i in 0..8 {
        let row_sum: f64 = (0..2).map(|c| proba[[i, c]]).sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-12,
            "predict_proba row {i} must sum to 1, got {row_sum}"
        );
        for c in 0..2 {
            assert!(
                (proba[[i, c]] - SK_PROBA[i][c]).abs() < 1e-6,
                "priors=[0.9,0.1] predict_proba[{i}][{c}]: sklearn {}, ferrolearn {}",
                SK_PROBA[i][c],
                proba[[i, c]]
            );
        }
    }
}

/// REQ-6 (#580), R-DEV-4 deviation: a wrong-length `priors` is rejected with
/// `FerroError::ShapeMismatch`. sklearn silently mis-indexes a wrong-length
/// `priors` (CPython/numpy footgun); ferrolearn eliminates it by checking the
/// length in `fit` against `n_classes` (here 2 classes, 3 priors).
#[test]
fn qda_priors_length_mismatch() {
    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            0.0, 0.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 2.5, 2.5, 3.0, 3.0, 3.0,
        ],
    )
    .unwrap();
    let y = Array1::from(vec![0, 0, 0, 0, 1, 1, 1, 1]);

    let res = QDA::<f64>::new()
        .with_priors(array![0.3, 0.3, 0.4])
        .fit(&x, &y);
    assert!(
        matches!(res, Err(FerroError::ShapeMismatch { .. })),
        "wrong-length priors must return ShapeMismatch (R-DEV-4)"
    );
}

// ===========================================================================
// REQ-11 (#584, partial) — scalings_ / rotations_ fitted attributes.
// ===========================================================================

/// REQ-11 (#584): the SVD formulation now materializes per-class `scalings_`
/// (`S²/(n_k-1)`) and `rotations_` (`Vtᵀ`). `scalings_` is sign-invariant and is
/// pinned to the live `m.scalings_` oracle directly. `rotations_` (the SVD right
/// singular vectors) is only defined up to a per-column sign, which differs
/// between LAPACK implementations (faer vs numpy), so it is verified through the
/// sign-invariant reconstruction `R·diag(S2)·Rᵀ`, which must equal sklearn's
/// `store_covariance=True` per-class `covariance_` (`discriminant_analysis.py:952`:
/// `cov = V·(S²/(n-1))·Vᵀ`).
///
/// Oracle (live sklearn 1.5.2):
/// ```text
/// python3 -W ignore -c "import numpy as np; \
///   from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as Q; \
///   X=np.array([[0.,0.],[1.,.5],[.5,1.],[1.,1.],[2.,2.],[3.,2.5],[2.5,3.],[3.,3.],\
///               [1.,3.],[2.,4.],[1.5,3.5],[2.,3.]]); \
///   y=np.array([0,0,0,0,1,1,1,1,2,2,2,2]); \
///   m=Q(store_covariance=True).fit(X,y); \
///   [print(s.tolist()) for s in m.scalings_]; \
///   [print(c.tolist()) for c in m.covariance_]"
/// ```
#[test]
fn qda_scalings_rotations() {
    // Live sklearn 1.5.2 `m.scalings_` (per class, length min(n_k, n_features)).
    const SK_SCALINGS: [[f64; 2]; 3] = [
        [0.37500000000000006, 0.08333333333333333],
        [0.37500000000000006, 0.08333333333333333],
        [0.3333333333333333, 0.12500000000000003],
    ];
    // Live sklearn 1.5.2 `m.covariance_` (store_covariance=True) = V·diag(S2)·Vᵀ.
    const SK_COV: [[f64; 4]; 3] = [
        [
            0.22916666666666669,
            0.14583333333333334,
            0.14583333333333337,
            0.22916666666666666,
        ],
        [
            0.22916666666666669,
            0.14583333333333334,
            0.14583333333333337,
            0.22916666666666666,
        ],
        [
            0.2291666666666667,
            0.10416666666666664,
            0.10416666666666664,
            0.2291666666666666,
        ],
    ];

    let fitted = QDA::<f64>::new().fit(&mc_x(), &mc_y()).unwrap();
    let scalings = fitted.scalings();
    let rotations = fitted.rotations();

    assert_eq!(scalings.len(), 3, "one scalings vector per class");
    assert_eq!(rotations.len(), 3, "one rotations matrix per class");

    for c in 0..3 {
        // scalings_ is sign-invariant — pin directly to the oracle.
        assert_eq!(scalings[c].len(), 2, "scalings[{c}] length");
        for j in 0..2 {
            assert!(
                (scalings[c][j] - SK_SCALINGS[c][j]).abs() < 1e-12,
                "scalings[{c}][{j}]: sklearn {}, ferrolearn {}",
                SK_SCALINGS[c][j],
                scalings[c][j]
            );
        }

        // rotations_ defined up to per-column sign; verify the sign-invariant
        // reconstruction R·diag(S2)·Rᵀ == sklearn's covariance_ (V·diag(S2)·Vᵀ).
        let r = rotations[c]; // (n_features=2, k=2)
        assert_eq!(r.dim(), (2, 2), "rotations[{c}] shape (n_features, k)");
        let s2 = scalings[c];
        let mut recon = [[0.0_f64; 2]; 2];
        for a in 0..2 {
            for b in 0..2 {
                let mut acc = 0.0;
                for j in 0..2 {
                    acc += r[[a, j]] * s2[j] * r[[b, j]];
                }
                recon[a][b] = acc;
            }
        }
        for a in 0..2 {
            for b in 0..2 {
                let exp = SK_COV[c][a * 2 + b];
                assert!(
                    (recon[a][b] - exp).abs() < 1e-12,
                    "R·diag(S2)·Rᵀ[{c}][{a}][{b}]: sklearn {}, ferrolearn {}",
                    exp,
                    recon[a][b]
                );
            }
        }
    }
}

// ===========================================================================
// REQ-9 (#582) — store_covariance + covariance_.
// ===========================================================================

/// REQ-9 (#582): `QDA::with_store_covariance(true)` materializes the per-class
/// `covariance_`, reconstructed from the REGULARIZED scalings as
/// `Rₖ · diag(scalingsₖ) · Rₖᵀ`, matching sklearn
/// `QuadraticDiscriminantAnalysis(store_covariance=True).covariance_`
/// (`discriminant_analysis.py:951-952` `cov.append(np.dot(S2 * Vt.T, Vt))`,
/// where `S2` is the regularized scalings; `:955-956` `self.covariance_ = cov`).
/// With `store_covariance=False` (the default) sklearn does not set the
/// attribute at all, so `covariance()` must be `None`.
///
/// Oracle (live sklearn 1.5.2):
/// ```text
/// python3 -W ignore -c "import numpy as np; \
///   from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as Q; \
///   X=np.array([[1.,1.],[1.5,1.2],[1.2,0.8],[1.1,1.3],[5.,5.],[5.5,4.8],[4.8,5.2]]); \
///   y=np.array([0,0,0,0,1,1,1]); \
///   m=Q(store_covariance=True).fit(X,y); \
///   [print(repr(c.tolist())) for c in m.covariance_]; \
///   print('default attr:', hasattr(Q().fit(X,y),'covariance_'))"
/// # class0 -> [[0.04666666666666665, 0.010000000000000014],
/// #            [0.01000000000000002, 0.049166666666666664]]
/// # class1 -> [[0.13000000000000014, -0.07000000000000012],
/// #            [-0.07000000000000012, 0.04000000000000009]]
/// # default attr: False  (covariance_ only exists under store_covariance=True)
/// ```
#[test]
fn qda_store_covariance() {
    // Live sklearn 1.5.2 `Q(store_covariance=True).covariance_` (per class,
    // (n_features, n_features), row-major).
    const SK_COV: [[f64; 4]; 2] = [
        [
            0.04666666666666665,
            0.010000000000000014,
            0.01000000000000002,
            0.049166666666666664,
        ],
        [
            0.13000000000000014,
            -0.07000000000000012,
            -0.07000000000000012,
            0.04000000000000009,
        ],
    ];

    let x = Array2::from_shape_vec(
        (7, 2),
        vec![
            1.0, 1.0, 1.5, 1.2, 1.2, 0.8, 1.1, 1.3, // class 0
            5.0, 5.0, 5.5, 4.8, 4.8, 5.2, // class 1
        ],
    )
    .unwrap();
    let y = Array1::from(vec![0, 0, 0, 0, 1, 1, 1]);

    let fitted = QDA::<f64>::new()
        .with_store_covariance(true)
        .fit(&x, &y)
        .unwrap();

    let cov = fitted
        .covariance()
        .expect("store_covariance=true must populate covariance_");
    assert_eq!(cov.len(), 2, "one covariance matrix per class");
    for c in 0..2 {
        assert_eq!(
            cov[c].dim(),
            (2, 2),
            "covariance[{c}] is (n_features, n_features)"
        );
        for a in 0..2 {
            for b in 0..2 {
                let exp = SK_COV[c][a * 2 + b];
                assert!(
                    (cov[c][[a, b]] - exp).abs() < 1e-9,
                    "store_covariance covariance_[{c}][{a}][{b}]: sklearn {}, ferrolearn {}",
                    exp,
                    cov[c][[a, b]]
                );
            }
        }
    }

    // store_covariance=false (the default and the explicit-false builder) ->
    // covariance_ is not set (None), matching sklearn (hasattr -> False).
    let fitted_default = QDA::<f64>::new().fit(&x, &y).unwrap();
    assert!(
        fitted_default.covariance().is_none(),
        "default (store_covariance=false) must leave covariance_ unset (None)"
    );
    let fitted_false = QDA::<f64>::new()
        .with_store_covariance(false)
        .fit(&x, &y)
        .unwrap();
    assert!(
        fitted_false.covariance().is_none(),
        "with_store_covariance(false) must leave covariance_ unset (None)"
    );
}

// ===========================================================================
// REQ-4 (#578) — predict_log_proba oracle pin.
// ===========================================================================

/// REQ-4 (#578): `predict_log_proba` is the elementwise log of `predict_proba`,
/// matching sklearn `QuadraticDiscriminantAnalysis().predict_log_proba`
/// (`discriminant_analysis.py:1058-1059` `probas_ = self.predict_proba(X);
/// return np.log(probas_)`). The shared OVERLAPPING 3-class `mc_x`/`mc_y`
/// dataset keeps every class probability in `(0, 1)`, so all log-probas are
/// FINITE (no `log(0) = -inf`) and an exact <1e-6 pin is meaningful.
///
/// Oracle (live sklearn 1.5.2):
/// ```text
/// python3 -W ignore -c "import numpy as np; \
///   from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as Q; \
///   X=np.array([[0.,0.],[1.,.5],[.5,1.],[1.,1.],[2.,2.],[3.,2.5],[2.5,3.],[3.,3.],\
///               [1.,3.],[2.,4.],[1.5,3.5],[2.,3.]]); \
///   y=np.array([0,0,0,0,1,1,1,1,2,2,2,2]); \
///   m=Q().fit(X,y); \
///   [print(repr(r)) for r in m.predict_log_proba(X).tolist()]"
/// ```
#[test]
fn qda_predict_log_proba() {
    // Live sklearn 1.5.2 `predict_log_proba(X)` (all entries finite).
    const SK_LOG_PROBA: [[f64; 3]; 12] = [
        [
            -2.970257239867861e-08,
            -17.333333363035898,
            -23.977174399261802,
        ],
        [
            -8.843091345757376e-05,
            -9.333421764246792,
            -18.664762800472694,
        ],
        [
            -9.701432257666628e-05,
            -9.333430347655908,
            -11.664771383881808,
        ],
        [
            -0.0012750479584154241,
            -6.667941714625081,
            -12.645116084184309,
        ],
        [
            -4.020637201309386,
            -0.020637201309386656,
            -5.997811570868613,
        ],
        [
            -12.000069628853462,
            -6.962885346522371e-05,
            -9.664743998412685,
        ],
        [
            -12.067311078830054,
            -0.0673110788300555,
            -2.7319854483892785,
        ],
        [
            -14.667968262678182,
            -0.001301596011517959,
            -6.645142632237407,
        ],
        [
            -16.022831745972976,
            -12.022831745972978,
            -6.115532202653518e-06,
        ],
        [
            -26.022837327601888,
            -11.356170660935224,
            -1.1697161113845318e-05,
        ],
        [
            -21.106166675711933,
            -11.772833342378599,
            -7.711937820823237e-06,
        ],
        [
            -11.263049363960029,
            -1.9297160306266947,
            -0.15689040018591827,
        ],
    ];

    let fitted = QDA::<f64>::new().fit(&mc_x(), &mc_y()).unwrap();
    let log_proba = fitted.predict_log_proba(&mc_x()).unwrap();

    assert_eq!(log_proba.dim(), (12, 3));
    for i in 0..12 {
        for c in 0..3 {
            assert!(
                log_proba[[i, c]].is_finite(),
                "predict_log_proba[{i}][{c}] must be finite (overlapping dataset)"
            );
            assert!(
                (log_proba[[i, c]] - SK_LOG_PROBA[i][c]).abs() < 1e-6,
                "predict_log_proba[{i}][{c}]: sklearn {}, ferrolearn {}",
                SK_LOG_PROBA[i][c],
                log_proba[[i, c]]
            );
        }
    }
}
