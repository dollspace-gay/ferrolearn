//! Divergence pins + value-contract guards for `BernoulliNB` /
//! `FittedBernoulliNB` (`ferrolearn-bayes/src/bernoulli.rs`) against the LIVE
//! scikit-learn 1.5.2 oracle (`from sklearn.naive_bayes import BernoulliNB`,
//! mirroring `sklearn/naive_bayes.py` `BernoulliNB` + `_BaseDiscreteNB`).
//!
//! Every expected value below is a LIVE `sklearn` 1.5.2 oracle value (computed
//! via `python3 -c "..."` run from `/tmp`, quoted above each block) — NEVER
//! copied from the ferrolearn side (goal.md R-CHAR-3).
//!
//! Mirrors `sklearn/naive_bayes.py`:
//! `BernoulliNB.__init__(*, alpha=1.0, force_alpha=True, binarize=0.0,
//! fit_prior=True, class_prior=None)` (naive_bayes.py:1159-1174) — `binarize`
//! DEFAULTS to `0.0`, not `None`; `_parameter_constraints` `binarize: [None,
//! Interval(Real, 0, None, closed="left")]` (naive_bayes.py:1156) and the
//! shared `alpha: [Interval(Real, 0, None, closed="left"), "array-like"]`
//! (naive_bayes.py:530, the `>= 0` HARD reject enforced at `fit` by
//! `_validate_params`); `_check_X_y` / `_check_X` binarize `X` only when
//! `binarize is not None` (naive_bayes.py:1176-1187); `_count` accumulates
//! `feature_count_ += Y.T @ X`, `class_count_ += Y.sum(axis=0)`
//! (naive_bayes.py:1189-1192); `_update_feature_log_prob` smoothed_fc=fc+alpha,
//! smoothed_cc=cc+alpha*2, feature_log_prob_=log(smoothed_fc)-log(smoothed_cc)
//! (naive_bayes.py:1194-1201); `_joint_log_likelihood` neg_prob=log(1-exp(flp)),
//! jll=X@(flp-neg).T + class_log_prior_ + neg.sum(axis=1)
//! (naive_bayes.py:1203-1219); `_update_class_log_prior` LENGTH-only check then
//! `log(class_prior)` (naive_bayes.py:580-602, NO sum/non-neg check for
//! discrete NB); `_check_alpha` floor 1e-10 unless `force_alpha`
//! (naive_bayes.py:604-626).
//!
//! Design doc: `.design/bayes/bernoulli.md` (commit 4d9e9938).
//!
//! Test taxonomy — RED pins FAIL now and go green when the generator lands the
//! fix; GREEN guards PASS now and protect the parts that are already correct:
//! `divergence_bernoulli_binarize_default_is_zero` (RED, #911);
//! `divergence_bernoulli_negative_alpha_rejected` (RED, #912);
//! `green_bernoulli_value_on_binary_data`,
//! `green_bernoulli_with_binarize_threshold_value`,
//! `green_bernoulli_class_prior_length_only`,
//! `green_bernoulli_score_accuracy` (GREEN).
//!
//! Documented-not-pinned (missing surface / multi-file):
//!
//! sample_weight on fit + partial_fit `classes=` (#908): sklearn `fit(X, y,
//! sample_weight=None)` (naive_bayes.py:712) weights the binarized `Y` so
//! `feature_count_ = Y.T @ X` / `class_count_ = Y.sum(axis=0)` become weighted
//! (naive_bayes.py:1189-1192) — `BernoulliNB().fit(Xbin, y,
//! sample_weight=[1,2,1,1,1,3]).feature_count_` = `[[4,2,0],[0,1,5]]`,
//! `class_count_` = `[4,5]`. ferrolearn's `Fit` trait is `fn fit(&self, x, y)` —
//! no `sample_weight` parameter on `fit` or `partial_fit`. The shared
//! `_BaseDiscreteNB.partial_fit(X, y, classes=None, ...)` (naive_bayes.py:629-708)
//! takes the full `classes` list on the first call and binarizes `y` against it,
//! so a NEW label in a later chunk is still represented; ferrolearn
//! `FittedBernoulliNB::partial_fit(&mut self, x, y)` loops only over the already
//! fitted `self.classes`, silently DROPPING a new later-chunk label. Both are
//! missing surface / multi-file API changes; no forced test here.
//!
//! fitted accessors feature_log_prob_/class_log_prior_/feature_count_/
//! class_count_ + the PyO3 surface (#909): sklearn exposes these
//! (naive_bayes.py:1088-1117); `FittedBernoulliNB` keeps
//! `log_prob`/`log_neg_prob`/`log_prior`/`feature_counts`/`class_counts` private
//! with no accessor (only `classes()` via `HasClasses`), and `_RsBernoulliNB`
//! (`ferrolearn-python/src/extras.rs`, the `py_classifier!` macro) exposes only
//! `new(alpha, fit_prior, binarize)` + `fit` + `predict` — no
//! `class_prior`/`force_alpha` kwargs, no `predict_proba`/`predict_log_proba`/
//! `predict_joint_log_proba`/`score`/`partial_fit` (which the library HAS), no
//! fitted-attr getters. NOTE: `BernoulliNB` exposes NO `coef_`/`intercept_` in
//! 1.5.2 (the deprecated `_BaseDiscreteNB` properties are gone;
//! `hasattr(BernoulliNB().fit(Xbin,y), 'coef_') == False`) — that is NOT a gap.
//! Missing surface; a pytest divergence belongs in ferrolearn-python.
//!
//! negative features (binarize=None): sklearn `BernoulliNB(binarize=None)` has
//! NO `check_non_negative` (UNLIKE MultinomialNB) — it ACCEPTS negatives (only
//! emitting a log-domain RuntimeWarning); ferrolearn `fn fit` also accepts
//! negatives → a MATCH on the accept decision. Documented, not pinned.
//!
//! ferray substrate (#910): `bernoulli.rs` imports `ndarray` + `num-traits`,
//! not `ferray-core`. Substrate migration; no observable-value pin.

use ferrolearn_bayes::BernoulliNB;
use ferrolearn_core::{Fit, Predict};
use ndarray::{Array1, Array2, array};

/// The shared already-BINARY 0/1 fixture from `.design/bayes/bernoulli.md`:
/// `Xbin = [[1,1,0],[1,0,0],[1,1,0],[0,0,1],[0,1,1],[0,0,1]]`,
/// `y = [0,0,0,1,1,1]`. On 0/1 data sklearn `binarize=0.0` and ferrolearn
/// `binarize=None` COINCIDE (`1>0→1`, `0>0→0` is identity on `{0,1}`;
/// `np.allclose(BernoulliNB().fit(Xbin,y).feature_log_prob_,
/// BernoulliNB(binarize=None).fit(Xbin,y).feature_log_prob_) == True`).
fn binary_fixture() -> (Array2<f64>, Array1<usize>) {
    let x = Array2::from_shape_vec(
        (6, 3),
        vec![
            1.0, 1.0, 0.0, // class 0
            1.0, 0.0, 0.0, // class 0
            1.0, 1.0, 0.0, // class 0
            0.0, 0.0, 1.0, // class 1
            0.0, 1.0, 1.0, // class 1
            0.0, 0.0, 1.0, // class 1
        ],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 1, 1, 1];
    (x, y)
}

/// The binary query `q = [[1,0,0],[0,1,1]]`.
fn binary_query() -> Array2<f64> {
    Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 1.0]).unwrap()
}

/// The NON-binary count fixture from `.design/bayes/bernoulli.md`:
/// `Xc = [[2,0,1],[0,3,0],[1,1,2],[0,0,4]]`, `yc = [0,0,1,1]`. With sklearn's
/// default `binarize=0.0`, every value `> 0` collapses to `1`; ferrolearn's
/// `binarize=None` keeps the raw counts — divergent `feature_count_` and
/// divergent `predict` labels.
fn nonbinary_fixture() -> (Array2<f64>, Array1<usize>) {
    let x = Array2::from_shape_vec(
        (4, 3),
        vec![
            2.0, 0.0, 1.0, // class 0
            0.0, 3.0, 0.0, // class 0
            1.0, 1.0, 2.0, // class 1
            0.0, 0.0, 4.0, // class 1
        ],
    )
    .unwrap();
    let y = array![0usize, 0, 1, 1];
    (x, y)
}

// ===========================================================================
// RED #911 — `binarize` DEFAULTS to `0.0`, so on NON-binary data sklearn
// binarizes at 0 by default; ferrolearn `new()` defaults `binarize=None` and
// uses RAW counts.
//
// sklearn `BernoulliNB.__init__(..., binarize=0.0, ...)` (naive_bayes.py:1164):
//     binarize=0.0,
// → `_check_X_y` / `_check_X` (naive_bayes.py:1176-1187):
//     if self.binarize is not None: X = binarize(X, threshold=self.binarize)
// so by DEFAULT every `fit`/`predict` binarizes `X` at threshold 0.0 (every
// value `> 0` → 1). ferrolearn `BernoulliNB::<F>::new()` (bernoulli.rs) sets
// `binarize: None`, so on NON-binary `X` `fn fit` skips binarization and
// computes raw count SUMS, not binary occurrence counts.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "from sklearn.naive_bayes import BernoulliNB; import numpy as np; \
//     Xc=np.array([[2.,0.,1.],[0.,3.,0.],[1.,1.,2.],[0.,0.,4.]]); yc=np.array([0,0,1,1]); \
//     m=BernoulliNB().fit(Xc,yc); \
//     print(BernoulliNB().binarize); print(m.feature_count_.tolist()); \
//     print(m.predict(Xc).tolist()); print(m.predict_proba(Xc).tolist())"
//   ->  binarize default      : 0.0
//   ->  feature_count_        : [[1.0, 1.0, 1.0], [1.0, 1.0, 2.0]]
//   ->  predict(Xc)           : [1, 0, 1, 1]
//   ->  predict_proba(Xc)     : [[0.39999999999999997, 0.5999999999999999],
//                                [0.6666666666666669, 0.3333333333333332],
//                                [0.39999999999999997, 0.5999999999999999],
//                                [0.39999999999999997, 0.5999999999999999]]
//
// ferrolearn `BernoulliNB::<f64>::new()` (binarize=None) instead computes raw
// feature sums `[[2,3,1],[1,1,6]]` and (simulating that path) `predict(Xc) =
// [1,1,0,1]` — labels DIFFER from sklearn `[1,0,1,1]` at rows 1 and 2 (rows
// 1,2 are the cleanest discriminator). We pin BOTH the labels and a
// predict_proba VALUE (which differs everywhere). FAILS now; goes green once
// `new()` defaults `binarize = Some(0.0)` mirroring naive_bayes.py:1164.
// ===========================================================================

/// Divergence: ferrolearn `BernoulliNB::new()` defaults `binarize = None`,
/// whereas sklearn `BernoulliNB.__init__` defaults `binarize=0.0`
/// (`sklearn/naive_bayes.py:1164`), so on NON-binary data sklearn binarizes at
/// 0 (occurrence indicators) while ferrolearn uses raw counts. On
/// `Xc=[[2,0,1],[0,3,0],[1,1,2],[0,0,4]]`, `yc=[0,0,1,1]`: sklearn
/// `predict(Xc)=[1,0,1,1]`; ferrolearn (binarize=None, raw counts) diverges at
/// rows 1 and 2. Tracking: #911
#[test]
fn divergence_bernoulli_binarize_default_is_zero() {
    let (xc, yc) = nonbinary_fixture();
    let fitted = BernoulliNB::<f64>::new().fit(&xc, &yc).unwrap();

    // sklearn 1.5.2 oracle (quoted above) — NOT copied from ferrolearn.
    let preds: Array1<usize> = fitted.predict(&xc).unwrap();
    assert_eq!(
        preds,
        array![1usize, 0, 1, 1],
        "BernoulliNB::new().predict(Xc) should match sklearn's default-binarize \
         result [1,0,1,1] (sklearn binarize=0.0, naive_bayes.py:1164); \
         ferrolearn binarize=None uses raw counts (#911)"
    );

    // Row 1 (=[0,3,0]) predict_proba VALUE under sklearn's default binarize.
    // sklearn predict_proba(Xc)[1] = [0.6666666666666669, 0.3333333333333332].
    let proba = fitted.predict_proba(&xc).unwrap();
    const SK_PROBA_ROW1: [f64; 2] = [0.666_666_666_666_666_9, 0.333_333_333_333_333_2];
    const TOL: f64 = 1e-9;
    for c in 0..2 {
        assert!(
            (proba[[1, c]] - SK_PROBA_ROW1[c]).abs() <= TOL,
            "predict_proba(Xc)[1][{c}]: ferrolearn {} vs sklearn {} \
             (sklearn default binarize=0.0, #911)",
            proba[[1, c]],
            SK_PROBA_ROW1[c]
        );
    }
}

// ===========================================================================
// RED #912 — `alpha >= 0` is a HARD reject at fit.
//
// sklearn `_BaseDiscreteNB._parameter_constraints` (naive_bayes.py:530),
// merged into `BernoulliNB._parameter_constraints` (naive_bayes.py:1154-1157):
//     "alpha": [Interval(Real, 0, None, closed="left"), "array-like"],
// → alpha must be >= 0, enforced at `fit` by `_validate_params`.
// `BernoulliNB(alpha=-0.5).fit(X,y)` raises `InvalidParameterError` (a
// ValueError subclass). This is DISTINCT from `_check_alpha`'s 1e-10 floor
// (naive_bayes.py:604-626), which only fires for alpha < 1e-10 when
// force_alpha=False.
//
// ferrolearn `fn fit` (bernoulli.rs) computes
//     let alpha = crate::clamp_alpha(self.alpha, self.force_alpha);
// where `clamp_alpha` (= base::check_alpha) only FLOORS under
// force_alpha=false. With the default force_alpha=true, `clamp_alpha(-0.5,
// true)` returns -0.5 unchanged, so `fit` proceeds and computes
// `log((count - 0.5)/denom)` — negative-smoothed garbage / NaN, NO error.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "from sklearn.naive_bayes import BernoulliNB; import numpy as np; \
//     Xc=np.array([[2.,0.,1.],[0.,3.,0.],[1.,1.,2.],[0.,0.,4.]]); yc=np.array([0,0,1,1]); \
//     BernoulliNB(alpha=-0.5).fit(Xc,yc)"
//   ->  InvalidParameterError :: The 'alpha' parameter of BernoulliNB must be a
//       float in the range [0.0, inf) or an array-like. Got -0.5 instead.
//
// We pin the OBSERVABLE: `fit` must return `Err` (sklearn raises). ferrolearn
// currently returns `Ok`: FAILS now. Minimally fixable in `bernoulli.rs`
// `fn fit` — reject `alpha < 0` (InvalidParameter) before/around clamp_alpha.
// ===========================================================================

/// Divergence: ferrolearn's `BernoulliNB::fit` accepts a negative `alpha`,
/// whereas `sklearn/naive_bayes.py:530`
/// (`alpha: Interval(Real, 0, None, closed="left")`) makes `alpha < 0` a HARD
/// reject at `fit` (`InvalidParameterError`). `BernoulliNB(alpha=-0.5).fit`
/// raises in sklearn; ferrolearn returns `Ok` because `clamp_alpha(-0.5, true)`
/// passes -0.5 through unchanged (the floor only fires under force_alpha=false).
/// Tracking: #912
#[test]
fn divergence_bernoulli_negative_alpha_rejected() {
    let (xc, yc) = nonbinary_fixture();

    // alpha = -0.5 — sklearn raises InvalidParameterError (oracle above).
    let model = BernoulliNB::<f64>::new().with_alpha(-0.5);
    let result = model.fit(&xc, &yc);

    assert!(
        result.is_err(),
        "fit with alpha=-0.5 should error (sklearn: InvalidParameterError, \
         'The 'alpha' parameter of BernoulliNB must be a float in the range \
         [0.0, inf) ...'), got Ok (#912)"
    );
}

// ===========================================================================
// GREEN — feature_log_prob_ smoothing + _joint_log_likelihood VALUE on
// ALREADY-BINARY data (REQ-1). On 0/1 data sklearn `binarize=0.0` and
// ferrolearn `binarize=None` COINCIDE (oracle `np.allclose == True`), so this
// isolates the jll/predict VALUE contract from the binarize-default issue
// (#911). Mirrors `_update_feature_log_prob` (naive_bayes.py:1194-1201) feeding
// `_joint_log_likelihood` neg_prob=log(1-exp(flp)); jll=X@(flp-neg).T +
// class_log_prior_ + neg.sum (naive_bayes.py:1203-1219) through _BaseNB.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "from sklearn.naive_bayes import BernoulliNB; import numpy as np; \
//     Xbin=np.array([[1.,1.,0.],[1.,0.,0.],[1.,1.,0.],[0.,0.,1.],[0.,1.,1.],[0.,0.,1.]]); \
//     y=np.array([0,0,0,1,1,1]); q=np.array([[1.,0.,0.],[0.,1.,1.]]); \
//     m=BernoulliNB().fit(Xbin,y); \
//     print(m.feature_log_prob_.tolist()); print(m.predict_proba(q).tolist()); \
//     print(m.predict_log_proba(q).tolist()); print(m.predict(q).tolist())"
//   -> feature_log_prob_  [[-0.2231435513142097, -0.5108256237659905, -1.6094379124341003],
//                          [-1.6094379124341003, -0.916290731874155, -0.2231435513142097]]
//   -> predict_proba(q)   [[0.9142857142857143, 0.08571428571428572],
//                          [0.08571428571428567, 0.9142857142857145]]
//   -> predict_log_proba(q) [[-0.0896121586896872, -2.456735772821304],
//                          [-2.4567357728213044, -0.08961215868968697]]
//   -> predict(q)         [0, 1]
// (feature_log_prob_ has no public accessor; verified indirectly via
// predict_proba / predict_log_proba / predict.)
// ===========================================================================

/// Guard: `predict_proba` / `predict_log_proba` / `predict` on the already-
/// BINARY fixture match sklearn `BernoulliNB().fit(Xbin,y)` to ~1e-12 (the
/// `_update_feature_log_prob` + Bernoulli `_joint_log_likelihood` VALUE,
/// naive_bayes.py:1194-1219). The REQ-1 value lock the doc-author flagged.
#[test]
fn green_bernoulli_value_on_binary_data() {
    let (x, y) = binary_fixture();
    let q = binary_query();
    let fitted = BernoulliNB::<f64>::new().fit(&x, &y).unwrap();

    let proba = fitted.predict_proba(&q).unwrap();
    let log_proba = fitted.predict_log_proba(&q).unwrap();
    let preds: Array1<usize> = fitted.predict(&q).unwrap();

    // sklearn 1.5.2 oracle (quoted above) — NOT copied from ferrolearn.
    const SK_PROBA: [[f64; 2]; 2] = [
        [0.914_285_714_285_714_3, 0.085_714_285_714_285_72],
        [0.085_714_285_714_285_67, 0.914_285_714_285_714_5],
    ];
    const SK_LOG_PROBA: [[f64; 2]; 2] = [
        [-0.089_612_158_689_687_2, -2.456_735_772_821_304],
        [-2.456_735_772_821_304_4, -0.089_612_158_689_686_97],
    ];
    const TOL: f64 = 1e-12;

    for i in 0..2 {
        for c in 0..2 {
            assert!(
                (proba[[i, c]] - SK_PROBA[i][c]).abs() <= TOL,
                "predict_proba[{i}][{c}]: ferrolearn {} vs sklearn {}",
                proba[[i, c]],
                SK_PROBA[i][c]
            );
            assert!(
                (log_proba[[i, c]] - SK_LOG_PROBA[i][c]).abs() <= TOL,
                "predict_log_proba[{i}][{c}]: ferrolearn {} vs sklearn {}",
                log_proba[[i, c]],
                SK_LOG_PROBA[i][c]
            );
        }
    }
    // sklearn 1.5.2 oracle: predict(q) == [0, 1].
    assert_eq!(preds, array![0usize, 1]);
}

// ===========================================================================
// GREEN — binarize threshold VALUE when `binarize` is SET (REQ-2). sklearn
// `binarize(X, threshold)` is strictly-greater (`X > threshold → 1`), applied
// in `_check_X_y`/`_check_X` when `binarize is not None`
// (naive_bayes.py:1176-1187). ferrolearn `binarize_array` is `v > threshold →
// 1 else 0` — also strictly-greater (NOT `>=`).
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "from sklearn.naive_bayes import BernoulliNB; import numpy as np; \
//     Xcont=np.array([[0.9,0.8,0.1],[0.7,0.2,0.3],[0.8,0.9,0.1],[0.2,0.1,0.9],[0.1,0.8,0.7],[0.3,0.2,0.8]]); \
//     y=np.array([0,0,0,1,1,1]); q=np.array([[0.9,0.1,0.2],[0.1,0.2,0.9]]); \
//     m=BernoulliNB(binarize=0.5).fit(Xcont,y); \
//     print(m.predict_proba(q).tolist()); print(m.predict(q).tolist())"
//   -> predict_proba(q)  [[0.9142857142857143, 0.08571428571428572],
//                         [0.03999999999999999, 0.9600000000000001]]
//   -> predict(q)        [0, 1]
// ===========================================================================

/// Guard: `with_binarize(0.5)` on continuous data matches
/// `BernoulliNB(binarize=0.5).fit(Xcont,y)` (strictly-greater `v>0.5`,
/// naive_bayes.py:1176-1187) — `predict_proba`/`predict` to ~1e-12.
#[test]
fn green_bernoulli_with_binarize_threshold_value() {
    let xcont = Array2::from_shape_vec(
        (6, 3),
        vec![
            0.9, 0.8, 0.1, // class 0
            0.7, 0.2, 0.3, // class 0
            0.8, 0.9, 0.1, // class 0
            0.2, 0.1, 0.9, // class 1
            0.1, 0.8, 0.7, // class 1
            0.3, 0.2, 0.8, // class 1
        ],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 1, 1, 1];
    let q = Array2::from_shape_vec((2, 3), vec![0.9, 0.1, 0.2, 0.1, 0.2, 0.9]).unwrap();

    let fitted = BernoulliNB::<f64>::new()
        .with_binarize(0.5)
        .fit(&xcont, &y)
        .unwrap();
    let proba = fitted.predict_proba(&q).unwrap();
    let preds: Array1<usize> = fitted.predict(&q).unwrap();

    // sklearn 1.5.2 oracle (quoted above) — NOT copied from ferrolearn.
    const SK_PROBA: [[f64; 2]; 2] = [
        [0.914_285_714_285_714_3, 0.085_714_285_714_285_72],
        [0.039_999_999_999_999_99, 0.960_000_000_000_000_1],
    ];
    const TOL: f64 = 1e-12;
    for i in 0..2 {
        for c in 0..2 {
            assert!(
                (proba[[i, c]] - SK_PROBA[i][c]).abs() <= TOL,
                "predict_proba[{i}][{c}]: ferrolearn {} vs sklearn {}",
                proba[[i, c]],
                SK_PROBA[i][c]
            );
        }
    }
    // sklearn 1.5.2 oracle: predict(q) == [0, 1].
    assert_eq!(preds, array![0usize, 1]);
}

// ===========================================================================
// GREEN — class_prior LENGTH-only validation (REQ-4, the MATCH, NOT a
// divergence). sklearn `_update_class_log_prior` (naive_bayes.py:589-591)
// checks ONLY length then `class_log_prior_ = np.log(class_prior)` — discrete
// NB has NO sum-to-1 and NO non-negativity check (UNLIKE GaussianNB). So
// `class_prior=[0.5,0.3]` (sum 0.8) is ACCEPTED on the sklearn side. ferrolearn
// `fn fit` checks ONLY length then `log_prior[ci] = p.ln()` — must ALSO accept.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "from sklearn.naive_bayes import BernoulliNB; import numpy as np; \
//     Xbin=np.array([[1.,1.,0.],[1.,0.,0.],[1.,1.,0.],[0.,0.,1.],[0.,1.,1.],[0.,0.,1.]]); \
//     y=np.array([0,0,0,1,1,1]); \
//     m=BernoulliNB(class_prior=[0.5,0.3]).fit(Xbin,y); \
//     print(m.class_log_prior_.tolist()); print(float(m.class_log_prior_[0]-m.class_log_prior_[1]))"
//   ->  class_log_prior_   [-0.6931471805599453, -1.2039728043259361]  (= log([0.5,0.3]); NO error)
//   ->  gap                0.5108256237659908
//
// The class-prior term enters `_joint_log_likelihood` ADDITIVELY. On this
// balanced fixture the empirical prior is log(0.5) for BOTH classes, so its
// inter-class log-prior gap is 0. Hence the observable
//   (jll_prior[i,0]-jll_prior[i,1]) - (jll_emp[i,0]-jll_emp[i,1])
// equals class_log_prior_[0]-class_log_prior_[1] = log(0.5)-log(0.3).
// ===========================================================================

/// Guard: ferrolearn `with_class_prior([0.5,0.3]).fit` SUCCEEDS (sum 0.8),
/// matching sklearn `_update_class_log_prior` LENGTH-only validation
/// (naive_bayes.py:589-591) — discrete NB has NO sum/non-neg check. The
/// resulting `class_log_prior_` = `log([0.5,0.3])`; we assert its inter-class
/// gap VALUE via joint-log-likelihood column differences (the only observable,
/// since `log_prior` is private).
#[test]
fn green_bernoulli_class_prior_length_only() {
    let (x, y) = binary_fixture();

    // sum 0.8 — sklearn accepts (oracle above). ferrolearn must too.
    let model = BernoulliNB::<f64>::new().with_class_prior(vec![0.5, 0.3]);
    let result = model.fit(&x, &y);
    assert!(
        result.is_ok(),
        "fit with class_prior=[0.5,0.3] (sum 0.8) should SUCCEED (discrete NB \
         has no sum check; sklearn class_log_prior_ = log([0.5,0.3]) = \
         [-0.6931..,-1.2040..]), got Err"
    );

    let fitted = model.fit(&x, &y).unwrap();
    let empirical = BernoulliNB::<f64>::new().fit(&x, &y).unwrap();
    let q = binary_query();
    let jll_prior = fitted.predict_joint_log_proba(&q).unwrap();
    let jll_emp = empirical.predict_joint_log_proba(&q).unwrap();

    // sklearn 1.5.2 oracle: class_log_prior_[0]-class_log_prior_[1] for
    // class_prior=[0.5,0.3] (= log(0.5)-log(0.3)); the balanced empirical prior
    // gap is 0, so the difference-of-gaps isolates this value.
    const SK_PRIOR_GAP: f64 = 0.510_825_623_765_990_8;
    const TOL: f64 = 1e-12;
    for i in 0..2 {
        let gap_prior = jll_prior[[i, 0]] - jll_prior[[i, 1]];
        let gap_emp = jll_emp[[i, 0]] - jll_emp[[i, 1]];
        let observed = gap_prior - gap_emp;
        assert!(
            (observed - SK_PRIOR_GAP).abs() <= TOL,
            "row {i}: explicit-vs-empirical prior gap = {observed}, want \
             class_log_prior_[0]-class_log_prior_[1] = {SK_PRIOR_GAP}"
        );
    }
}

// ===========================================================================
// GREEN — score (mean accuracy) on the separable binary fixture (the
// `ClassifierMixin.score` analog).
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "from sklearn.naive_bayes import BernoulliNB; import numpy as np; \
//     Xbin=np.array([[1.,1.,0.],[1.,0.,0.],[1.,1.,0.],[0.,0.,1.],[0.,1.,1.],[0.,0.,1.]]); \
//     y=np.array([0,0,0,1,1,1]); print(BernoulliNB().fit(Xbin,y).score(Xbin,y))"
//   ->  1.0
// ===========================================================================

/// Guard: `score(X,y)` matches sklearn `BernoulliNB().fit(Xbin,y).score(Xbin,y)`
/// = `1.0` on this separable binary fixture.
#[test]
fn green_bernoulli_score_accuracy() {
    let (x, y) = binary_fixture();
    let fitted = BernoulliNB::<f64>::new().fit(&x, &y).unwrap();

    let acc = fitted.score(&x, &y).unwrap();

    // sklearn 1.5.2 oracle: 1.0.
    const SK_SCORE: f64 = 1.0;
    assert!(
        (acc - SK_SCORE).abs() <= 1e-12,
        "score = {acc}, want {SK_SCORE}"
    );
}

// ===========================================================================
// RED (#2106) — `binarize = NaN / +inf` parameter validation. Follow-up to the
// #2105 fix, which added a `b < F::zero()` reject in `BernoulliNB::fit`
// (bernoulli.rs:257-264) mirroring ONLY the lower bound of sklearn's
//   `binarize: [None, Interval(Real, 0, None, closed="left")]`
//   (sklearn/naive_bayes.py:1156).
// IEEE makes that guard INCOMPLETE: `NaN < 0.0 == false` and `+inf < 0.0 ==
// false`, so ferrolearn ACCEPTS both and fits. sklearn's half-open interval
// [0.0, +inf) excludes NaN (all comparisons false) and excludes +inf (the OPEN
// right end, `closed="left"`), so `_validate_params()` raises
// `InvalidParameterError` (a `ValueError`) for BOTH.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "import numpy as np; from sklearn.naive_bayes import BernoulliNB
//   X=np.array([[2.,0.],[0.,3.],[1.,1.],[0.,4.]]); y=np.array([0,0,1,1])
//   for b in (float('nan'), float('inf')):
//       try: BernoulliNB(binarize=b).fit(X,y); print(b,'OK')
//       except Exception as e: print(b, type(e).__name__)"
//   ->  nan InvalidParameterError
//       inf InvalidParameterError
// ===========================================================================

/// Divergence: ferrolearn `BernoulliNB::fit` accepts `binarize = NaN` where
/// sklearn rejects it. sklearn's `Interval(Real, 0, None, closed="left")`
/// (`sklearn/naive_bayes.py:1156`) excludes NaN, so `_validate_params()` raises
/// `InvalidParameterError`. ferrolearn's guard `b < F::zero()` (bernoulli.rs:257)
/// evaluates `NaN < 0.0 == false`, so the fit succeeds. Tracking: #2106
#[test]
#[ignore = "divergence: BernoulliNB::fit accepts NaN binarize, sklearn rejects (naive_bayes.py:1156); tracking #2106"]
fn divergence_bernoulli_nan_binarize_rejected() {
    let (xc, yc) = nonbinary_fixture();

    let model = BernoulliNB::<f64>::new().with_binarize(f64::NAN);
    let result = model.fit(&xc, &yc);

    assert!(
        result.is_err(),
        "fit with binarize=NaN should error (sklearn: InvalidParameterError, \
         NaN outside Interval[0.0, inf)), got Ok (#2106)"
    );
}

/// Divergence: ferrolearn `BernoulliNB::fit` accepts `binarize = +inf` where
/// sklearn rejects it. sklearn's `Interval(Real, 0, None, closed="left")`
/// (`sklearn/naive_bayes.py:1156`) has an OPEN upper bound, so +inf is excluded
/// and `_validate_params()` raises `InvalidParameterError`. ferrolearn's guard
/// `b < F::zero()` (bernoulli.rs:257) evaluates `+inf < 0.0 == false`, so the fit
/// succeeds. (`-inf` is correctly rejected by both.) Tracking: #2106
#[test]
#[ignore = "divergence: BernoulliNB::fit accepts +inf binarize, sklearn rejects (naive_bayes.py:1156); tracking #2106"]
fn divergence_bernoulli_inf_binarize_rejected() {
    let (xc, yc) = nonbinary_fixture();

    let model = BernoulliNB::<f64>::new().with_binarize(f64::INFINITY);
    let result = model.fit(&xc, &yc);

    assert!(
        result.is_err(),
        "fit with binarize=+inf should error (sklearn: InvalidParameterError, \
         +inf is the open right end of Interval[0.0, inf)), got Ok (#2106)"
    );
}
