//! Divergence pins for `LinearDiscriminantAnalysis` against the live
//! scikit-learn 1.5.2 oracle (`sklearn/discriminant_analysis.py`, commit
//! 156ef14).
//!
//! LDA is DETERMINISTIC (no RNG — closed-form SVD/eigensolve + Bayes rule), so
//! exact value parity is testable. sklearn's DEFAULT `solver="svd"` whitens the
//! within-class data, derives `xbar_`/`scalings_`, then forms an AFFINE
//! classifier: `transform(X) = (X - xbar_) @ scalings_`
//! (`discriminant_analysis.py:685`), `decision_function(X) = X @ coef_.T +
//! intercept_` (the `LinearClassifierMixin`, referenced at `:739`), where
//! `intercept_` embeds `log(priors_)` (`:557`). `predict` is the argmax of that
//! affine decision; `predict_proba` is its softmax (`:706-711`).
//!
//! ferrolearn implements a FUNDAMENTALLY DIFFERENT algorithm (the design doc
//! `.design/linear/lda.md` records this): the classical `Sw⁻¹·Sb` Fisher
//! eigensolve. Its `transform` is the raw, un-centered, un-whitened
//! `X @ scalings` (`lda.rs::transform`); its `decision_function`/`predict`/
//! `predict_proba` are a nearest-centroid / EQUAL-PRIORS approximation
//! `-½‖z − μ_c‖²` in the projected space (`lda.rs::decision_function`,
//! `lda.rs::predict`). So the core inference paths diverge in VALUE, and on
//! imbalanced classes the missing `log(priors_)` term flips labels.
//!
//! Every expected value below is produced by RUNNING scikit-learn 1.5.2 (the
//! live oracle), never copied from ferrolearn (goal.md R-CHAR-3). The exact
//! python invocation that produced each constant is recorded in a comment.
//!
//! Tracking: #588 (decision_function / svd fit), #589 (predict argmax /
//! imbalanced priors), #592 (transform), #593 (provided `priors`),
//! #591 (`predict_log_proba`).

use ferrolearn_core::traits::{Fit, Predict, Transform};
use ferrolearn_linear::LDA;
use ferrolearn_linear::lda::{Shrinkage, Solver};
use ndarray::{Array1, Array2, array};

// ===========================================================================
// Balanced 3-class / 2-feature dataset for transform + decision_function.
// Uniform priors (4 per class) so the ONLY remaining differences are the
// algorithm-level ones (centering/whitening for transform; affine-vs-distance
// for decision_function) — NOT the prior term.
//
//   X = [[0,0],[1,.5],[.5,1],[1,1],         class 0
//        [4,4],[5,4.5],[4.5,5],[5,5],        class 1
//        [0,5],[1,6],[.5,5.5],[1,5]]         class 2
//   y = [0,0,0,0, 1,1,1,1, 2,2,2,2]
// ===========================================================================

fn mc_x() -> Array2<f64> {
    Array2::from_shape_vec(
        (12, 2),
        vec![
            0.0, 0.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, // class 0
            4.0, 4.0, 5.0, 4.5, 4.5, 5.0, 5.0, 5.0, // class 1
            0.0, 5.0, 1.0, 6.0, 0.5, 5.5, 1.0, 5.0, // class 2
        ],
    )
    .unwrap()
}

fn mc_y() -> Array1<usize> {
    Array1::from(vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
}

/// Divergence (#592, REQ-5): sklearn's svd-solver `transform` is the centered,
/// whitened projection `(X - xbar_) @ scalings_`
/// (`discriminant_analysis.py:685`). ferrolearn's `FittedLDA::transform`
/// returns the raw, un-centered, un-whitened `X @ scalings`
/// (`lda.rs::transform`), so the values diverge in scale AND offset — this is
/// NOT a per-column sign flip (the comparison below allows a per-column sign
/// and still fails).
///
/// Oracle (live sklearn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; \
///   from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as L; \
///   X=np.array([[0.,0.],[1.,.5],[.5,1.],[1.,1.],[4.,4.],[5.,4.5],[4.5,5.],[5.,5.],\
///               [0.,5.],[1.,6.],[.5,5.5],[1.,5.]]); \
///   y=np.array([0,0,0,0,1,1,1,1,2,2,2,2]); \
///   m=L().fit(X,y); print(repr(m.transform(X).tolist()))"
/// ```
#[test]
fn lda_transform_parity() {
    // Live sklearn 1.5.2 `transform(X)` (shape (12, 2)).
    const SK_TRANSFORM: [[f64; 2]; 12] = [
        [-4.72428385678745, 5.697246226728315],
        [-5.512647175227892, 3.7531314720812357],
        [-3.2838815095491394, 4.1720118253429845],
        [-4.289591170922204, 3.384346789373375],
        [-2.9855131133264674, -3.5543515226914444],
        [-3.77387643176691, -5.498466277338524],
        [-1.545110766088157, -5.079585924076775],
        [-2.5508204274612223, -5.867250960046385],
        [7.506276186269428, 2.009399399649709],
        [7.940968872134674, -0.3035000377052312],
        [7.72362252920205, 0.8529496809722388],
        [5.494856863523298, 0.43406932771049006],
    ];

    let fitted = LDA::<f64>::new(Some(2)).fit(&mc_x(), &mc_y()).unwrap();
    let tr = fitted.transform(&mc_x()).unwrap();

    assert_eq!(tr.dim(), (12, 2), "transform shape");

    // sklearn's SVD scalings are defined up to a per-column sign, so compare
    // each column allowing a global sign flip. The divergence is real if EVEN
    // the best-sign match fails.
    for col in 0..2 {
        // Pick the sign that best aligns column `col` to the oracle (robust to
        // the discriminant-direction sign ambiguity), then assert match.
        let mut agree = 0i32;
        for i in 0..12 {
            if (tr[[i, col]] - SK_TRANSFORM[i][col]).abs()
                <= (tr[[i, col]] + SK_TRANSFORM[i][col]).abs()
            {
                agree += 1;
            }
        }
        let sign = if agree >= 6 { 1.0 } else { -1.0 };
        for i in 0..12 {
            assert!(
                (sign * tr[[i, col]] - SK_TRANSFORM[i][col]).abs() < 1e-6,
                "transform[{i}][{col}] (sign-aligned): sklearn {}, ferrolearn {} \
                 (raw {})",
                SK_TRANSFORM[i][col],
                sign * tr[[i, col]],
                tr[[i, col]]
            );
        }
    }
}

/// Divergence (#588, REQ-1): sklearn's `decision_function(X)` is the affine
/// `X @ coef_.T + intercept_` (`discriminant_analysis.py:739`, derived from
/// `_solve_svd` at `:556-559`), whose `intercept_` carries `log(priors_)`.
/// ferrolearn's `FittedLDA::decision_function` returns `-½‖z − μ_c‖²` in the
/// projected space (`lda.rs::decision_function`) — a DIFFERENT function. Even
/// with uniform priors the per-sample VALUES disagree (only the argmax happens
/// to coincide on balanced data).
///
/// Oracle (live sklearn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; \
///   from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as L; \
///   X=np.array([[0.,0.],[1.,.5],[.5,1.],[1.,1.],[4.,4.],[5.,4.5],[4.5,5.],[5.,5.],\
///               [0.,5.],[1.,6.],[.5,5.5],[1.,5.]]); \
///   y=np.array([0,0,0,0,1,1,1,1,2,2,2,2]); \
///   m=L().fit(X,y); print(repr(m.decision_function(X).tolist()))"
/// ```
#[test]
fn lda_decision_function_parity() {
    // Live sklearn 1.5.2 `decision_function(X)` (shape (12, 3), multiclass).
    const SK_DEC: [[f64; 3]; 12] = [
        [25.208393205837393, -32.94545294800878, -56.65081009086592],
        [20.452898700342885, -21.085562838118673, -63.75520569526151],
        [12.310041557485741, -29.228419980975808, -47.46949140954723],
        [13.439162436606619, -22.56083756339339, -55.26619470625052],
        [-21.868529871085702, 8.593008590452762, -51.11234855240434],
        [-26.624024376580206, 20.452898700342864, -58.21674415679993],
        [-34.76688151943735, 12.310041557485732, -41.93102987108566],
        [-33.63776064031647, 18.97762397506814, -49.72773316778895],
        [-44.928969431525246, -47.69820020075599, 28.23929979924398],
        [-56.69820020075602, -37.313584816140605, 29.623915183859367],
        [-50.81358481614063, -42.5058925084483, 28.931607491551674],
        [-42.670727673283494, -34.36303536559116, 12.645893205837382],
    ];

    let fitted = LDA::<f64>::new(Some(2)).fit(&mc_x(), &mc_y()).unwrap();
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

// ===========================================================================
// Imbalanced 2-class / 2-feature dataset for the prior-flip divergence.
// class 0: 20 points clustered near the origin; class 1: 2 points near (4,0).
// Empirical priors => [0.9091, 0.0909]. The borderline EVAL points are
// appended after the training rows so the predict() output has known indices.
//
// On the borderline point (2.0, 0.0): sklearn's affine decision carries
// log(priors_) = log([0.9091, 0.0909]), which shifts the boundary TOWARD the
// minority class — i.e. it takes a stronger signal to call class 1 — so
// sklearn predicts 0. ferrolearn's equal-priors nearest-centroid ignores the
// prior and assigns the point to whichever projected centroid is closer.
// ===========================================================================

fn imb_train() -> (Array2<f64>, Array1<usize>) {
    let x = Array2::from_shape_vec(
        (22, 2),
        vec![
            // class 0 (20 points near origin)
            -1.0, 0.2, -0.5, -0.3, 0.0, 0.1, 0.5, -0.1, 1.0, 0.4, -0.8, -0.2, 0.3, 0.3, -0.2, -0.4,
            0.7, 0.0, -0.6, 0.5, 0.1, -0.5, -0.4, 0.2, 0.9, -0.3, -0.9, 0.1, 0.2, 0.4, -0.1, -0.2,
            0.6, 0.3, -0.7, -0.1, 0.4, 0.0, -0.3, 0.2, // class 1 (2 points near (4,0))
            3.6, 0.1, 4.4, -0.1,
        ],
    )
    .unwrap();
    let y = Array1::from(vec![
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
    ]);
    (x, y)
}

fn imb_eval() -> Array2<f64> {
    // Three borderline points along the discriminant axis.
    Array2::from_shape_vec((3, 2), vec![2.0, 0.0, 2.2, 0.0, 1.8, 0.0]).unwrap()
}

/// Divergence (#589, REQ-2): on an IMBALANCED 2-class set (empirical priors
/// `[0.9091, 0.0909]`), sklearn's `predict` = argmax of the affine
/// `X @ coef_.T + intercept_` whose `intercept_` carries `log(priors_)`
/// (`discriminant_analysis.py:557`, `:739`). The prior term shifts the boundary
/// toward the minority class, so the borderline point `(2.0, 0.0)` is predicted
/// class 0. ferrolearn's `FittedLDA::predict` is the EQUAL-PRIORS
/// nearest-centroid in the projected space (`lda.rs::predict`), which ignores
/// `log(priors_)` and can disagree.
///
/// Oracle (live sklearn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; \
///   from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as L; \
///   c0=[(-1.,.2),(-.5,-.3),(0.,.1),(.5,-.1),(1.,.4),(-.8,-.2),(.3,.3),(-.2,-.4),\
///       (.7,0.),(-.6,.5),(.1,-.5),(-.4,.2),(.9,-.3),(-.9,.1),(.2,.4),(-.1,-.2),\
///       (.6,.3),(-.7,-.1),(.4,0.),(-.3,.2)]; \
///   c1=[(3.6,.1),(4.4,-.1)]; \
///   X=np.array([list(p) for p in c0+c1]); y=np.array([0]*20+[1]*2); \
///   m=L().fit(X,y); print('priors_', m.priors_.round(8).tolist()); \
///   Xe=np.array([[2.,0.],[2.2,0.],[1.8,0.]]); \
///   print('predict_eval', m.predict(Xe).tolist()); \
///   print('predict_proba_eval', repr(m.predict_proba(Xe).tolist()))"
/// # priors_ [0.90909091, 0.09090909]
/// # predict_eval [0, 1, 0]
/// # predict_proba_eval [[0.8878604503843278, 0.11213954961567214],
/// #                     [0.46705576494273016, 0.5329442350572698],
/// #                     [0.9862124956481378, 0.013787504351862137]]
/// ```
#[test]
fn lda_imbalanced_priors_predict() {
    // Live sklearn 1.5.2 `predict([[2,0],[2.2,0],[1.8,0]])` on the imbalanced
    // set. The prior shifts (2.0, 0.0) -> class 0.
    const SK_PRED_EVAL: [usize; 3] = [0, 1, 0];
    // Live sklearn 1.5.2 `predict_proba` of the same eval points. The
    // probabilities embed log(priors_), so they differ from ferrolearn's
    // equal-prior softmax regardless of whether the argmax label happens to
    // agree.
    const SK_PROBA_EVAL: [[f64; 2]; 3] = [
        [0.8878604503843278, 0.11213954961567214],
        [0.46705576494273016, 0.5329442350572698],
        [0.9862124956481378, 0.013787504351862137],
    ];

    let (x, y) = imb_train();
    let fitted = LDA::<f64>::new(None).fit(&x, &y).unwrap();
    let xe = imb_eval();

    let pred = fitted.predict(&xe).unwrap();
    assert_eq!(pred.len(), 3);
    for i in 0..3 {
        assert_eq!(
            pred[i], SK_PRED_EVAL[i],
            "imbalanced predict_eval[{i}]: sklearn {}, ferrolearn {} \
             (sklearn's log(priors_) shifts the boundary toward the minority \
             class; ferrolearn's equal-prior nearest-centroid ignores it)",
            SK_PRED_EVAL[i], pred[i]
        );
    }

    // Even where the argmax label agrees, the prior-aware probabilities differ
    // from ferrolearn's equal-prior softmax of the projected distances.
    let proba = fitted.predict_proba(&xe).unwrap();
    assert_eq!(proba.dim(), (3, 2));
    for i in 0..3 {
        for c in 0..2 {
            assert!(
                (proba[[i, c]] - SK_PROBA_EVAL[i][c]).abs() < 1e-6,
                "imbalanced predict_proba[{i}][{c}]: sklearn {}, ferrolearn {}",
                SK_PROBA_EVAL[i][c],
                proba[[i, c]]
            );
        }
    }
}

// ===========================================================================
// #593 — provided `priors` (LDA::with_priors).
//
// sklearn's constructor `priors` (default `None`,
// `discriminant_analysis.py:351,359`) feeds `priors_` directly when given
// (`:605`, `self.priors_ = xp.asarray(self.priors)`, used VERBATIM — no
// renormalization when it already sums to 1). The provided priors flow into
// `xbar_ = priors_ @ means_` (`:517`), the between-class scaling
// `sqrt(n·priors_·fac)` (`:540`), and `intercept_ += log(priors_)` (`:557`),
// shifting the affine decision relative to the empirical default.
// ===========================================================================

fn pp_x() -> Array2<f64> {
    Array2::from_shape_vec(
        (8, 2),
        vec![
            0.0, 0.0, 1.0, 1.0, 2.0, 0.5, 0.5, 2.0, // class 0
            5.0, 5.0, 6.0, 4.5, 4.5, 6.0, 5.5, 5.5, // class 1
        ],
    )
    .unwrap()
}

fn pp_y() -> Array1<usize> {
    Array1::from(vec![0, 0, 0, 0, 1, 1, 1, 1])
}

/// #593 (REQ-7): provided `priors=[0.9, 0.1]` are used VERBATIM as `priors_`
/// and shift the decision (via `xbar_` + `log(priors_)`) relative to the
/// empirical `[0.5, 0.5]` default. The `predict_proba` matches the live oracle
/// to 1e-6, `priors()` returns the provided vector verbatim, and the empirical
/// default differs.
///
/// Oracle (live sklearn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; \
///   from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as L; \
///   X=np.array([[0.,0.],[1.,1.],[2.,0.5],[0.5,2.],[5.,5.],[6.,4.5],[4.5,6.],[5.5,5.5]]); \
///   y=np.array([0,0,0,0,1,1,1,1]); \
///   print(repr(L(priors=[0.9,0.1]).fit(X,y).predict_proba(X).tolist()))"
/// # [[1.0, 4.4173717141944625e-31], [1.0, 1.759372325813511e-21],
/// #  [1.0, 4.419836550435889e-19], [1.0, 4.419836550435889e-19],
/// #  [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
/// ```
/// The empirical default `priors_ = [0.5, 0.5]` (verified live by `L().fit(...)`)
/// produces a strictly different `priors_` vector.
#[test]
fn lda_provided_priors() {
    // Live sklearn 1.5.2 predict_proba with priors=[0.9, 0.1] (shape (8, 2)).
    const SK_PROBA: [[f64; 2]; 8] = [
        [1.0, 4.4173717141944625e-31],
        [1.0, 1.759372325813511e-21],
        [1.0, 4.419836550435889e-19],
        [1.0, 4.419836550435889e-19],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ];

    let x = pp_x();
    let y = pp_y();

    let fitted = LDA::<f64>::new(None)
        .with_priors(array![0.9, 0.1])
        .fit(&x, &y)
        .unwrap();

    // priors_ is the provided vector VERBATIM (sklearn :605).
    let priors = fitted.priors();
    assert_eq!(priors.len(), 2);
    assert!(
        (priors[0] - 0.9).abs() < 1e-12,
        "priors_[0] = {}",
        priors[0]
    );
    assert!(
        (priors[1] - 0.1).abs() < 1e-12,
        "priors_[1] = {}",
        priors[1]
    );

    // predict_proba matches the live oracle to 1e-6 (shape-stable (n, 2)).
    let proba = fitted.predict_proba(&x).unwrap();
    assert_eq!(proba.dim(), (8, 2), "predict_proba shape");
    for i in 0..8 {
        for c in 0..2 {
            assert!(
                (proba[[i, c]] - SK_PROBA[i][c]).abs() < 1e-6,
                "provided-priors predict_proba[{i}][{c}]: sklearn {}, ferrolearn {}",
                SK_PROBA[i][c],
                proba[[i, c]]
            );
        }
    }

    // The empirical default (priors=None) resolves a DIFFERENT priors_
    // ([0.5, 0.5], the live sklearn `L().fit(X,y).priors_`).
    let empirical = LDA::<f64>::new(None).fit(&x, &y).unwrap();
    let emp = empirical.priors();
    assert!(
        (emp[0] - 0.5).abs() < 1e-12 && (emp[1] - 0.5).abs() < 1e-12,
        "empirical priors_ should be [0.5, 0.5], got [{}, {}]",
        emp[0],
        emp[1]
    );
    assert!(
        (emp[0] - priors[0]).abs() > 1e-3,
        "empirical priors_ must differ from the provided [0.9, 0.1]"
    );
}

/// #603 (REQ-7): sklearn LDA `fit` rejects negative provided priors
/// (`discriminant_analysis.py:607-608`, `if xp.any(self.priors_ < 0): raise
/// ValueError("priors must be non-negative")`). UNLIKE QDA, LDA validates the
/// provided `priors_`. ferrolearn previously used them verbatim; it must now
/// `Err`.
///
/// Oracle (live sklearn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; \
///   from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as L; \
///   X=np.array([[0.,0.],[1.,1.],[2.,0.5],[0.5,2.],[5.,5.],[6.,4.5],[4.5,6.],[5.5,5.5]]); \
///   y=np.array([0,0,0,0,1,1,1,1]); L(priors=[-0.1,1.1]).fit(X,y)"
/// # ValueError: priors must be non-negative
/// ```
#[test]
fn lda_priors_negative_rejected() {
    let x = pp_x();
    let y = pp_y(); // 2 classes
    let result = LDA::<f64>::new(None)
        .with_priors(array![-0.1, 1.1])
        .fit(&x, &y);
    assert!(
        result.is_err(),
        "negative provided priors must error (sklearn raises ValueError, \
         discriminant_analysis.py:607-608)"
    );
}

/// #603 (REQ-7): sklearn LDA `fit` renormalizes the provided priors when they
/// do not sum to 1 (`discriminant_analysis.py:610-612`, `if
/// xp.abs(xp.sum(self.priors_) - 1.0) > 1e-5: warnings.warn(...); self.priors_
/// = self.priors_ / self.priors_.sum()`). With `priors=[0.5, 0.6]` (sum 1.1),
/// `priors_` becomes `[0.4545..., 0.5454...]` and `predict_proba` matches the
/// live oracle (which renormalizes internally) to 1e-6.
///
/// Oracle (live sklearn 1.5.2):
/// ```text
/// python3 -c "import numpy as np,warnings; \
///   from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as L; \
///   X=np.array([[0.,0.],[1.,1.],[2.,0.5],[0.5,2.],[5.,5.],[6.,4.5],[4.5,6.],[5.5,5.5]]); \
///   y=np.array([0,0,0,0,1,1,1,1]); warnings.simplefilter('ignore'); \
///   m=L(priors=[0.5,0.6]).fit(X,y); \
///   print(repr(m.priors_.tolist()), repr(m.predict_proba(X).tolist()))"
/// # [0.45454545454545453, 0.5454545454545454]
/// # [[1.0, 4.77e-30], [1.0, 1.90e-20], [1.0, 4.77e-18], [1.0, 4.77e-18],
/// #  [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
/// ```
#[test]
fn lda_priors_renormalized() {
    // Live sklearn 1.5.2 renormalized `priors_` for the provided [0.5, 0.6].
    const SK_PRIORS: [f64; 2] = [0.45454545454545453, 0.5454545454545454];
    // Live sklearn 1.5.2 predict_proba with priors=[0.5, 0.6] (renormalized).
    const SK_PROBA: [[f64; 2]; 8] = [
        [1.0, 4.7707614513299825e-30],
        [1.0, 1.900122111878563e-20],
        [1.0, 4.773423474470688e-18],
        [1.0, 4.773423474470688e-18],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ];

    let x = pp_x();
    let y = pp_y();

    let fitted = LDA::<f64>::new(None)
        .with_priors(array![0.5, 0.6])
        .fit(&x, &y)
        .unwrap();

    // priors_ is renormalized to sum 1 (sklearn :610-612).
    let priors = fitted.priors();
    assert_eq!(priors.len(), 2);
    for c in 0..2 {
        assert!(
            (priors[c] - SK_PRIORS[c]).abs() < 1e-12,
            "renormalized priors_[{c}]: sklearn {}, ferrolearn {}",
            SK_PRIORS[c],
            priors[c]
        );
    }

    // predict_proba matches the live oracle (which renormalizes internally).
    let proba = fitted.predict_proba(&x).unwrap();
    assert_eq!(proba.dim(), (8, 2), "predict_proba shape");
    for i in 0..8 {
        for c in 0..2 {
            assert!(
                (proba[[i, c]] - SK_PROBA[i][c]).abs() < 1e-6,
                "renormalized predict_proba[{i}][{c}]: sklearn {}, ferrolearn {}",
                SK_PROBA[i][c],
                proba[[i, c]]
            );
        }
    }
}

/// #593 (REQ-7, R-DEV-4): a `priors` vector whose length does not match the
/// number of classes is rejected (sklearn would silently mis-index it when
/// computing `xbar_`/`intercept_`, `:517,540,557`). `with_priors([0.3,0.3,0.4])`
/// on a 2-class dataset → `Err`.
#[test]
fn lda_provided_priors_length_mismatch() {
    let x = pp_x();
    let y = pp_y(); // 2 classes
    let result = LDA::<f64>::new(None)
        .with_priors(array![0.3, 0.3, 0.4])
        .fit(&x, &y);
    assert!(
        result.is_err(),
        "3-element priors on a 2-class dataset must error"
    );
}

// ===========================================================================
// #591 — predict_log_proba oracle pin.
//
// sklearn's `predict_log_proba` (`discriminant_analysis.py:713-737`) is
// `predict_proba` followed by an exact-zero `smallest_normal` floor and
// elementwise `log`. On an OVERLAPPING multiclass dataset every probability is
// strictly positive, so every log-proba is finite and the floor never fires —
// a clean 1e-6 assertion against the live oracle.
// ===========================================================================

fn lp_x() -> Array2<f64> {
    Array2::from_shape_vec(
        (12, 2),
        vec![
            0.0, 0.0, 1.0, 0.5, 0.5, 1.0, 2.0, 1.5, // class 0
            2.0, 2.0, 3.0, 1.8, 1.8, 2.5, 2.5, 3.0, // class 1
            1.0, 3.0, 2.0, 4.0, 0.5, 3.5, 3.0, 2.0, // class 2
        ],
    )
    .unwrap()
}

fn lp_y() -> Array1<usize> {
    Array1::from(vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
}

/// #591 (REQ-4): `predict_log_proba(X) = log(predict_proba(X))` with sklearn's
/// exact-zero `smallest_normal` floor (`discriminant_analysis.py:713-737`). On
/// this overlapping 3-class set all log-probas are finite; match the live
/// oracle to 1e-6.
///
/// Oracle (live sklearn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; \
///   from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as L; \
///   X=np.array([[0.,0.],[1.,.5],[.5,1.],[2.,1.5],[2.,2.],[3.,1.8],[1.8,2.5],[2.5,3.],\
///               [1.,3.],[2.,4.],[.5,3.5],[3.,2.]]); \
///   y=np.array([0,0,0,0,1,1,1,1,2,2,2,2]); \
///   print(repr(L().fit(X,y).predict_log_proba(X).tolist()))"
/// ```
#[test]
fn lda_predict_log_proba() {
    // Live sklearn 1.5.2 predict_log_proba(X) (shape (12, 3), all finite).
    const SK_LOG_PROBA: [[f64; 3]; 12] = [
        [
            -0.0001587412147669112,
            -8.81424374538817,
            -11.500273228176736,
        ],
        [
            -0.007767261965175309,
            -4.925071263751486,
            -7.6522673234724765,
        ],
        [
            -0.017383613399066335,
            -4.260126861781577,
            -5.772207198980196,
        ],
        [
            -1.2117694284515876,
            -0.48365825811924473,
            -2.455665938735218,
        ],
        [-2.8079420405217523, -0.331196700457846, -1.50684942503638],
        [
            -4.035961820544284,
            -0.11052331571747627,
            -2.4422395556808483,
        ],
        [-4.400938867709207, -0.6051887244448422, -0.8169821863919637],
        [-7.740817660189232, -0.6927305643344344, -0.6944341433230202],
        [-5.336182619023403, -1.510315772151896, -0.2557370516856885],
        [-11.60682696215518, -2.135544943165018, -0.1257778435937946],
        [-7.052209041503329, -2.551781441228021, -0.08208699823944263],
        [-4.758871118466485, -0.1339789457470513, -2.1471532032954483],
    ];

    let fitted = LDA::<f64>::new(Some(2)).fit(&lp_x(), &lp_y()).unwrap();
    let log_proba = fitted.predict_log_proba(&lp_x()).unwrap();

    assert_eq!(log_proba.dim(), (12, 3), "predict_log_proba shape");
    for i in 0..12 {
        // The max log-proba per row is <= 0 (it is log of a probability).
        let row_max = (0..3)
            .map(|c| log_proba[[i, c]])
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            row_max <= 1e-12,
            "row {i} max log-proba {row_max} should be <= 0"
        );
        for c in 0..3 {
            assert!(
                log_proba[[i, c]].is_finite(),
                "predict_log_proba[{i}][{c}] should be finite"
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

// ===========================================================================
// #598 (REQ-12) — store_covariance / covariance_.
//
// sklearn's svd-solver stores the SHARED within-class covariance only when
// `store_covariance=True` (default `False`, `discriminant_analysis.py:353`):
// `if self.store_covariance: self.covariance_ = _class_cov(X, y, self.priors_)`
// (`:509-510`). `_class_cov` (`:128-172`) returns
// `Σ_k priors_[k] · cov(X_k)`, where `cov(X_k)` is the MAXIMUM-LIKELIHOOD
// empirical covariance of class k's samples — `empirical_covariance` calls
// `np.cov(Xg.T, bias=1)` (`covariance/_empirical_covariance.py:109`),
// i.e. normalized by `n_k` (NOT `n_k - 1`). When `store_covariance=False`
// (the default), the attribute does not exist; ferrolearn returns `None`.
// ===========================================================================

fn cov_x() -> Array2<f64> {
    Array2::from_shape_vec(
        (8, 2),
        vec![
            0.0, 0.0, 1.0, 1.0, 2.0, 0.5, 0.5, 2.0, // class 0
            5.0, 5.0, 6.0, 4.5, 4.5, 6.0, 5.5, 5.5, // class 1
        ],
    )
    .unwrap()
}

fn cov_y() -> Array1<usize> {
    Array1::from(vec![0, 0, 0, 0, 1, 1, 1, 1])
}

/// #598 (REQ-12): `LDA::with_store_covariance(true)` populates
/// `FittedLDA::covariance` with the shared within-class covariance
/// `Σ_k priors_[k] · cov(X_k)` (`discriminant_analysis.py:509-510`, `_class_cov`
/// `:128-172`). Per-class `cov(X_k)` is the maximum-likelihood empirical
/// covariance (`np.cov(..., bias=1)`, ÷`n_k`). The default / `false` path leaves
/// `covariance() == None` (sklearn only defines the attribute when the flag is
/// set).
///
/// Oracle (live sklearn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; \
///   from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as L; \
///   X=np.array([[0.,0.],[1.,1.],[2.,.5],[.5,2.],[5.,5.],[6.,4.5],[4.5,6.],[5.5,5.5]]); \
///   y=np.array([0,0,0,0,1,1,1,1]); \
///   print(repr(L(store_covariance=True).fit(X,y).covariance_.tolist()))"
/// # [[0.4296875, -0.1328125], [-0.1328125, 0.4296875]]
/// ```
#[test]
fn lda_store_covariance() {
    // Live sklearn 1.5.2 covariance_ for store_covariance=True (shape (2, 2)).
    const SK_COV: [[f64; 2]; 2] = [[0.4296875, -0.1328125], [-0.1328125, 0.4296875]];

    let x = cov_x();
    let y = cov_y();

    // store_covariance=True ⇒ covariance() is Some and matches the oracle.
    let fitted = LDA::<f64>::new(None)
        .with_store_covariance(true)
        .fit(&x, &y)
        .unwrap();
    let cov = fitted
        .covariance()
        .expect("covariance() must be Some when store_covariance=true");
    assert_eq!(cov.dim(), (2, 2), "covariance_ shape");
    for a in 0..2 {
        for b in 0..2 {
            assert!(
                (cov[[a, b]] - SK_COV[a][b]).abs() < 1e-9,
                "covariance_[{a}][{b}]: sklearn {}, ferrolearn {}",
                SK_COV[a][b],
                cov[[a, b]]
            );
        }
    }

    // store_covariance=false (explicit) ⇒ covariance() is None.
    let fitted_false = LDA::<f64>::new(None)
        .with_store_covariance(false)
        .fit(&x, &y)
        .unwrap();
    assert!(
        fitted_false.covariance().is_none(),
        "covariance() must be None when store_covariance=false"
    );

    // Default (sklearn `store_covariance=False`, :353) ⇒ covariance() is None.
    let fitted_default = LDA::<f64>::new(None).fit(&x, &y).unwrap();
    assert!(
        fitted_default.covariance().is_none(),
        "covariance() must be None by default (sklearn store_covariance=False)"
    );
}

// ===========================================================================
// #601 (REQ-15) — tol parameter.
//
// sklearn's svd-solver `tol` (constructor default `1e-4`,
// `discriminant_analysis.py:354`, validated `Interval(Real, 0, None)` :343)
// drives BOTH SVD rank thresholds: `rank = sum(S > self.tol)` (`:532`) and
// `rank2 = sum(S > self.tol * S[0])` (`:554`). ferrolearn previously hardcoded
// `1e-4`; it is now a constructor parameter (`LDA::with_tol`) threaded into both
// rank cutoffs. The default `1e-4` keeps the svd fit byte-identical (the other
// oracle tests in this file remain green), so the structural assertion below
// (default value + plumb-through) plus the unchanged fit oracles fully pin it.
// ===========================================================================

/// #601 (REQ-15): `LDA::tol` defaults to sklearn's `1e-4`
/// (`discriminant_analysis.py:354`) and `with_tol` sets it. The default value is
/// the sklearn `file:line` symbolic constant (the constructor default at `:354`,
/// not a value copied from ferrolearn).
#[test]
fn lda_tol_param() {
    // sklearn constructor default `tol=1e-4` (discriminant_analysis.py:354).
    const SK_TOL_DEFAULT: f64 = 1e-4;

    // Default tol == sklearn's 1e-4.
    let lda = LDA::<f64>::new(None);
    assert!(
        (lda.tol() - SK_TOL_DEFAULT).abs() < 1e-18,
        "default tol must be sklearn's 1e-4, got {}",
        lda.tol()
    );

    // with_tol sets the field.
    let lda = lda.with_tol(0.5);
    assert!(
        (lda.tol() - 0.5).abs() < 1e-18,
        "with_tol(0.5) must set tol, got {}",
        lda.tol()
    );

    // Default tol still produces a valid fit (byte-identical svd path: the
    // other oracle tests assert exact values; here we confirm the threaded
    // default does not change the fitted shape).
    let x = cov_x();
    let y = cov_y();
    let fitted = LDA::<f64>::new(None).fit(&x, &y).unwrap();
    assert_eq!(fitted.scalings().nrows(), 2);
}

// ===========================================================================
// #595 (REQ-9) — lsqr solver / #597 (REQ-11) — shrinkage.
//
// sklearn's `solver="lsqr"` (`_solve_lstsq`, discriminant_analysis.py:365-419)
// computes `covariance_ = _class_cov(X, y, priors_, shrinkage)` (`:413`, the
// shrinkage-aware `Σ_k priors_[k] · cov(X_k)`, _class_cov :128-172, _cov
// :36-93), `coef_ = lstsq(covariance_, means_.T)[0].T` (`:416`), and
// `intercept_ = -0.5*diag(means_ @ coef_.T) + log(priors_)` (`:417-418`).
//
// For the binary case sklearn COLLAPSES coef_/intercept_ to a single row
// `coef_[1] - coef_[0]` (`:651-657`). ferrolearn (matching its existing svd
// path, open prereq blocker #600) keeps the FULL `(n_classes, n_features)`
// coef_, so the assertions below reconstruct the collapsed discriminant
// `coef_[1] - coef_[0]` and compare it to the live (collapsed) oracle.
//
// The shared dataset (X=cov_x(), y=cov_y()):
//   X = [[0,0],[1,1],[2,0.5],[0.5,2],[5,5],[6,4.5],[4.5,6],[5.5,5.5]]
//   y = [0,0,0,0,1,1,1,1]
// ===========================================================================

/// #595 (REQ-9): the `lsqr` solver matches the live sklearn 1.5.2
/// `LinearDiscriminantAnalysis(solver='lsqr').fit(X,y)` (no shrinkage). The
/// collapsed discriminant `coef_[1] - coef_[0]` matches `coef_` (already
/// collapsed by sklearn) to 1e-6, and `predict`/`predict_proba` match the
/// oracle.
///
/// Oracle (live sklearn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; \
///   from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as L; \
///   X=np.array([[0.,0.],[1.,1.],[2.,.5],[.5,2.],[5.,5.],[6.,4.5],[4.5,6.],[5.5,5.5]]); \
///   y=np.array([0,0,0,0,1,1,1,1]); m=L(solver='lsqr').fit(X,y); \
///   print(repr(m.coef_.tolist()), m.predict(X).tolist(), repr(m.predict_proba(X).tolist()))"
/// # coef_ [[14.736842105263158, 14.736842105263154]]
/// # predict [0, 0, 0, 0, 1, 1, 1, 1]
/// # predict_proba [[1.0, 6.30e-40], [1.0, 3.98e-27], [1.0, 6.30e-24],
/// #   [1.0, 6.30e-24], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
/// ```
#[test]
fn lda_lsqr_solver() {
    // Live sklearn 1.5.2 (collapsed binary) `coef_` for solver='lsqr'.
    const SK_COEF_COLLAPSED: [f64; 2] = [14.736842105263158, 14.736842105263154];
    const SK_PRED: [usize; 8] = [0, 0, 0, 0, 1, 1, 1, 1];
    const SK_PROBA: [[f64; 2]; 8] = [
        [1.0, 6.2980862979674404e-40],
        [1.0, 3.976189014922218e-27],
        [1.0, 6.3027724011859036e-24],
        [1.0, 6.3027724011859036e-24],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ];

    let x = cov_x();
    let y = cov_y();
    let fitted = LDA::<f64>::new(None)
        .with_solver(Solver::Lsqr)
        .fit(&x, &y)
        .unwrap();

    // coef_ is FULL (2, 2); the collapsed binary discriminant is row1 - row0.
    let coef = fitted.coef();
    assert_eq!(coef.dim(), (2, 2), "lsqr coef_ shape (uncollapsed)");
    for j in 0..2 {
        let collapsed = coef[[1, j]] - coef[[0, j]];
        assert!(
            (collapsed - SK_COEF_COLLAPSED[j]).abs() < 1e-6,
            "lsqr coef_[1]-coef_[0] [{j}]: sklearn {}, ferrolearn {}",
            SK_COEF_COLLAPSED[j],
            collapsed
        );
    }

    // covariance_ is ALWAYS populated for lsqr (sklearn :413).
    assert!(
        fitted.covariance().is_some(),
        "lsqr always sets covariance_ (sklearn :413)"
    );

    let pred = fitted.predict(&x).unwrap();
    for i in 0..8 {
        assert_eq!(pred[i], SK_PRED[i], "lsqr predict[{i}]");
    }

    let proba = fitted.predict_proba(&x).unwrap();
    assert_eq!(proba.dim(), (8, 2));
    for i in 0..8 {
        for c in 0..2 {
            assert!(
                (proba[[i, c]] - SK_PROBA[i][c]).abs() < 1e-6,
                "lsqr predict_proba[{i}][{c}]: sklearn {}, ferrolearn {}",
                SK_PROBA[i][c],
                proba[[i, c]]
            );
        }
    }
}

/// #597 (REQ-11): the `lsqr` solver with a FIXED shrinkage `0.5`
/// (`shrunk_covariance(emp_cov, 0.5)`, _shrunk_covariance.py:153-156:
/// `(1-s)*emp + s*(trace(emp)/p)*I`). The collapsed discriminant matches the
/// live oracle to 1e-6.
///
/// Oracle (live sklearn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; \
///   from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as L; \
///   X=np.array([[0.,0.],[1.,1.],[2.,.5],[.5,2.],[5.,5.],[6.,4.5],[4.5,6.],[5.5,5.5]]); \
///   y=np.array([0,0,0,0,1,1,1,1]); \
///   print(repr(L(solver='lsqr', shrinkage=0.5).fit(X,y).coef_.tolist()))"
/// # [[12.043010752688168, 12.043010752688172]]
/// ```
#[test]
fn lda_shrinkage_fixed() {
    const SK_COEF_COLLAPSED: [f64; 2] = [12.043010752688168, 12.043010752688172];

    let x = cov_x();
    let y = cov_y();
    let fitted = LDA::<f64>::new(None)
        .with_solver(Solver::Lsqr)
        .with_shrinkage(Shrinkage::Fixed(0.5))
        .fit(&x, &y)
        .unwrap();

    let coef = fitted.coef();
    assert_eq!(coef.dim(), (2, 2));
    for j in 0..2 {
        let collapsed = coef[[1, j]] - coef[[0, j]];
        assert!(
            (collapsed - SK_COEF_COLLAPSED[j]).abs() < 1e-6,
            "fixed-shrinkage coef_[1]-coef_[0] [{j}]: sklearn {}, ferrolearn {}",
            SK_COEF_COLLAPSED[j],
            collapsed
        );
    }
}

/// #597 (REQ-11): the `lsqr` solver with `shrinkage='auto'` (Ledoit-Wolf,
/// _cov :70-75 → ledoit_wolf_shrinkage, _shrunk_covariance.py:299-402). This
/// validates the analytical Ledoit-Wolf transcription against the live oracle.
///
/// Oracle (live sklearn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; \
///   from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as L; \
///   X=np.array([[0.,0.],[1.,1.],[2.,.5],[.5,2.],[5.,5.],[6.,4.5],[4.5,6.],[5.5,5.5]]); \
///   y=np.array([0,0,0,0,1,1,1,1]); \
///   print(repr(L(solver='lsqr', shrinkage='auto').fit(X,y).coef_.tolist()))"
/// # [[11.370558375634516, 11.370558375634513]]
/// ```
#[test]
fn lda_shrinkage_auto() {
    const SK_COEF_COLLAPSED: [f64; 2] = [11.370558375634516, 11.370558375634513];

    let x = cov_x();
    let y = cov_y();
    let fitted = LDA::<f64>::new(None)
        .with_solver(Solver::Lsqr)
        .with_shrinkage(Shrinkage::Auto)
        .fit(&x, &y)
        .unwrap();

    let coef = fitted.coef();
    assert_eq!(coef.dim(), (2, 2));
    for j in 0..2 {
        let collapsed = coef[[1, j]] - coef[[0, j]];
        assert!(
            (collapsed - SK_COEF_COLLAPSED[j]).abs() < 1e-6,
            "auto-shrinkage coef_[1]-coef_[0] [{j}]: sklearn {}, ferrolearn {} \
             (Ledoit-Wolf transcription)",
            SK_COEF_COLLAPSED[j],
            collapsed
        );
    }
}

// ===========================================================================
// #596 (REQ-10) — eigen solver (generalized-eigh via Cholesky reduction).
//
// sklearn's `solver="eigen"` (`_solve_eigen`, discriminant_analysis.py:421-485)
// computes `Sw = covariance_ = _class_cov(X, y, priors_, shrinkage)` (within
// scatter), `St = _cov(X, shrinkage)` (total scatter of the WHOLE X),
// `Sb = St - Sw`, then solves the GENERALIZED symmetric-definite eigenproblem
// `evals, evecs = linalg.eigh(Sb, Sw)` (`:475`); `explained_variance_ratio_ =
// sort(evals/sum(evals))[::-1][:max_components]` (`:476-478`); `evecs` sorted
// by DESCENDING eigenvalue (`:479`); `scalings_ = evecs`; `coef_ =
// (means_@evecs)@evecs.T` (`:482`); `intercept_ = -0.5*diag(means_@coef_.T) +
// log(priors_)` (`:483-485`).
//
// For the binary case sklearn COLLAPSES coef_/intercept_ to a single row
// `coef_[1] - coef_[0]` (`:651-657`). ferrolearn (matching its existing svd/
// lsqr paths, open prereq blocker #600) keeps the FULL `(n_classes,
// n_features)` coef_, so the assertions reconstruct the collapsed discriminant
// `coef_[1] - coef_[0]` and compare it to the live (collapsed) oracle. The
// collapsed coef_ and `coef_ = (means_@evecs)@evecs.T` are SIGN/ORDER-invariant
// under the eigenvector sign/order ambiguity, so they match regardless.
//
// Shared dataset (X=cov_x(), y=cov_y()):
//   X = [[0,0],[1,1],[2,0.5],[0.5,2],[5,5],[6,4.5],[4.5,6],[5.5,5.5]]
//   y = [0,0,0,0,1,1,1,1]
// ===========================================================================

/// #596 (REQ-10): the `eigen` solver matches the live sklearn 1.5.2
/// `LinearDiscriminantAnalysis(solver='eigen').fit(X,y)` (no shrinkage). The
/// collapsed discriminant `coef_[1] - coef_[0]` matches `coef_` (already
/// collapsed by sklearn) to 1e-6, `explained_variance_ratio_` is `[1.0]`, and
/// `predict`/`predict_proba` match the oracle.
///
/// Oracle (live sklearn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; \
///   from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as L; \
///   X=np.array([[0.,0.],[1.,1.],[2.,.5],[.5,2.],[5.,5.],[6.,4.5],[4.5,6.],[5.5,5.5]]); \
///   y=np.array([0,0,0,0,1,1,1,1]); m=L(solver='eigen').fit(X,y); \
///   print(repr(m.coef_.tolist()), repr(m.explained_variance_ratio_.tolist())); \
///   print(m.predict(X).tolist(), repr(m.predict_proba(X).tolist()))"
/// # coef_ [[14.736842105263165, 14.736842105263165]]
/// # explained_variance_ratio_ [1.0]
/// # predict [0, 0, 0, 0, 1, 1, 1, 1]
/// # predict_proba [[1.0, 6.298086297966993e-40], [1.0, 3.976189014921992e-27],
/// #   [1.0, 6.3027724011855906e-24], [1.0, 6.3027724011855906e-24],
/// #   [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
/// ```
#[test]
fn lda_eigen_solver() {
    // Live sklearn 1.5.2 (collapsed binary) `coef_` for solver='eigen'.
    const SK_COEF_COLLAPSED: [f64; 2] = [14.736842105263165, 14.736842105263165];
    // Live sklearn 1.5.2 explained_variance_ratio_ (truncated to max_components=1).
    const SK_EVR: [f64; 1] = [1.0];
    const SK_PRED: [usize; 8] = [0, 0, 0, 0, 1, 1, 1, 1];
    const SK_PROBA: [[f64; 2]; 8] = [
        [1.0, 6.298086297966993e-40],
        [1.0, 3.976189014921992e-27],
        [1.0, 6.3027724011855906e-24],
        [1.0, 6.3027724011855906e-24],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ];

    let x = cov_x();
    let y = cov_y();
    let fitted = LDA::<f64>::new(None)
        .with_solver(Solver::Eigen)
        .fit(&x, &y)
        .unwrap();

    // coef_ is FULL (2, 2); the collapsed binary discriminant is row1 - row0
    // (SIGN/ORDER-invariant under the eigenvector ambiguity).
    let coef = fitted.coef();
    assert_eq!(coef.dim(), (2, 2), "eigen coef_ shape (uncollapsed)");
    for j in 0..2 {
        let collapsed = coef[[1, j]] - coef[[0, j]];
        assert!(
            (collapsed - SK_COEF_COLLAPSED[j]).abs() < 1e-6,
            "eigen coef_[1]-coef_[0] [{j}]: sklearn {}, ferrolearn {}",
            SK_COEF_COLLAPSED[j],
            collapsed
        );
    }

    // explained_variance_ratio_ matches the oracle (truncated to max_components).
    let evr = fitted.explained_variance_ratio();
    assert_eq!(evr.len(), 1, "eigen evr length = max_components (1)");
    assert!(
        (evr[0] - SK_EVR[0]).abs() < 1e-9,
        "eigen explained_variance_ratio_[0]: sklearn {}, ferrolearn {}",
        SK_EVR[0],
        evr[0]
    );

    // covariance_ (= Sw) is ALWAYS populated for eigen (sklearn :467-469).
    assert!(
        fitted.covariance().is_some(),
        "eigen always sets covariance_ (sklearn :467-469)"
    );

    let pred = fitted.predict(&x).unwrap();
    for i in 0..8 {
        assert_eq!(pred[i], SK_PRED[i], "eigen predict[{i}]");
    }

    let proba = fitted.predict_proba(&x).unwrap();
    assert_eq!(proba.dim(), (8, 2));
    for i in 0..8 {
        for c in 0..2 {
            assert!(
                (proba[[i, c]] - SK_PROBA[i][c]).abs() < 1e-6,
                "eigen predict_proba[{i}][{c}]: sklearn {}, ferrolearn {}",
                SK_PROBA[i][c],
                proba[[i, c]]
            );
        }
    }
}

/// #596 (REQ-10): the `eigen` solver with a FIXED shrinkage `0.5` (shrinkage
/// is applied to BOTH `Sw` and `St`, `discriminant_analysis.py:467-472`). The
/// collapsed discriminant matches the live oracle to 1e-6.
///
/// Oracle (live sklearn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; \
///   from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as L; \
///   X=np.array([[0.,0.],[1.,1.],[2.,.5],[.5,2.],[5.,5.],[6.,4.5],[4.5,6.],[5.5,5.5]]); \
///   y=np.array([0,0,0,0,1,1,1,1]); \
///   print(repr(L(solver='eigen', shrinkage=0.5).fit(X,y).coef_.tolist()))"
/// # [[12.043010752688174, 12.04301075268817]]
/// ```
#[test]
fn lda_eigen_shrinkage() {
    const SK_COEF_COLLAPSED: [f64; 2] = [12.043010752688174, 12.04301075268817];

    let x = cov_x();
    let y = cov_y();
    let fitted = LDA::<f64>::new(None)
        .with_solver(Solver::Eigen)
        .with_shrinkage(Shrinkage::Fixed(0.5))
        .fit(&x, &y)
        .unwrap();

    let coef = fitted.coef();
    assert_eq!(coef.dim(), (2, 2));
    for j in 0..2 {
        let collapsed = coef[[1, j]] - coef[[0, j]];
        assert!(
            (collapsed - SK_COEF_COLLAPSED[j]).abs() < 1e-6,
            "eigen fixed-shrinkage coef_[1]-coef_[0] [{j}]: sklearn {}, ferrolearn {}",
            SK_COEF_COLLAPSED[j],
            collapsed
        );
    }
}

/// #597 (REQ-11): sklearn rejects `shrinkage` combined with the `svd` solver
/// (`discriminant_analysis.py:628-629`, `raise NotImplementedError("shrinkage
/// not supported with 'svd' solver.")`). ferrolearn's [`Fit::fit`] must return
/// an `Err` for `Solver::Svd` + any non-`None` shrinkage.
///
/// Oracle (live sklearn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; \
///   from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as L; \
///   X=np.array([[0.,0.],[1.,1.],[2.,.5],[.5,2.],[5.,5.],[6.,4.5],[4.5,6.],[5.5,5.5]]); \
///   y=np.array([0,0,0,0,1,1,1,1]); L(solver='svd', shrinkage=0.5).fit(X,y)"
/// # NotImplementedError: shrinkage not supported with 'svd' solver.
/// ```
#[test]
fn lda_svd_shrinkage_rejected() {
    let x = cov_x();
    let y = cov_y();
    let result = LDA::<f64>::new(None)
        .with_solver(Solver::Svd)
        .with_shrinkage(Shrinkage::Fixed(0.5))
        .fit(&x, &y);
    assert!(
        result.is_err(),
        "svd + shrinkage must error (sklearn NotImplementedError, :628-629)"
    );
}
